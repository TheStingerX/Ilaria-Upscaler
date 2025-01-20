import gradio as gr
import cv2
import numpy
import os
import random
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from torchvision.transforms.functional import rgb_to_grayscale

last_file = None
img_mode = "RGBA"

def realesrgan(img, model_name, denoise_strength, face_enhance, outscale):
    """Real-ESRGAN function to restore (and upscale) images.
    """
    if not img:
        return

    # Define model parameters
    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # Determine model paths
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Use dni to control the denoise strength
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # Restorer Class
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
        gpu_id=None
    )

    # Use GFPGAN for face enhancement
    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # Convert the input PIL image to cv2 image, so that it can be processed by realesrgan
    cv_img = numpy.array(img)
    img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)

    # Apply restoration
    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        # Save restored image and return it to the output Image component
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'

        out_filename = f"output_{rnd_string(8)}.{extension}"
        cv2.imwrite(out_filename, output)
        global last_file
        last_file = out_filename
        return out_filename


def rnd_string(x):
    """Returns a string of 'x' random characters
    """
    characters = "abcdefghijklmnopqrstuvwxyz_0123456789"
    result = "".join((random.choice(characters)) for i in range(x))
    return result


def reset():
    """Resets the Image components of the Gradio interface and deletes
    the last processed image
    """
    global last_file
    if last_file:
        print(f"Deleting {last_file} ...")
        os.remove(last_file)
        last_file = None
    return gr.update(value=None), gr.update(value=None)


def has_transparency(img):
    """This function works by first checking to see if a "transparency" property is defined
    in the image's info -- if so, we return "True". Then, if the image is using indexed colors
    (such as in GIFs), it gets the index of the transparent color in the palette
    (img.info.get("transparency", -1)) and checks if it's used anywhere in the canvas
    (img.getcolors()). If the image is in RGBA mode, then presumably it has transparency in
    it, but it double-checks by getting the minimum and maximum values of every color channel
    (img.getextrema()), and checks if the alpha channel's smallest value falls below 255.
    https://stackoverflow.com/questions/43864101/python-pil-check-if-image-is-transparent
    """
    if img.info.get("transparency", None) is not None:
        return True
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True
    return False


def image_properties(img):
    """Returns the dimensions (width and height) and color mode of the input image and
    also sets the global img_mode variable to be used by the realesrgan function
    """
    global img_mode
    if img:
        if has_transparency(img):
            img_mode = "RGBA"
        else:
            img_mode = "RGB"
        properties = f"Resolution: Width: {img.size[0]}, Height: {img.size[1]}  |  Color Mode: {img_mode}"
        return properties


def main():
    # Gradio Interface
    with gr.Blocks(theme=gr.themes.Default(primary_hue="pink", secondary_hue="rose"), title="Ilaria Upscaler 💖") as app:

        gr.Markdown(
            """# <div align="center"> Ilaria Upscaler 💖 </div>  
        """
        )

        with gr.Accordion("Upscaling option"):
            with gr.Row():
                model_name = gr.Dropdown(label="Upscaler model",
                                         choices=["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B",
                                                  "RealESRGAN_x2plus", "realesr-general-x4v3"],
                                         value="RealESRGAN_x4plus_anime_6B", show_label=True)
                denoise_strength = gr.Slider(label="Denoise Strength",
                                             minimum=0, maximum=1, step=0.1, value=0.5)
                outscale = gr.Slider(label="Resolution upscale",
                                     minimum=1, maximum=6, step=1, value=4, show_label=True)
                face_enhance = gr.Checkbox(label="Face Enhancement (GFPGAN)",
                )
                
        with gr.Row():
            with gr.Group():
                input_image = gr.Image(label="Input Image", type="pil", image_mode="RGBA")
                input_image_properties = gr.Textbox(label="Image Properties", max_lines=1)
            output_image = gr.Image(label="Output Image", image_mode="RGBA")
        with gr.Row():
            reset_btn = gr.Button("Remove images")
            restore_btn = gr.Button("Upscale")

        # Event listeners:
        input_image.change(fn=image_properties, inputs=input_image, outputs=input_image_properties)
        restore_btn.click(fn=realesrgan,
                          inputs=[input_image, model_name, denoise_strength, face_enhance, outscale],
                          outputs=output_image)
        reset_btn.click(fn=reset, inputs=[], outputs=[output_image, input_image])
        # reset_btn.click(None, inputs=[], outputs=[input_image], _js="() => (null)\n")
        # Undocumented method to clear a component's value using Javascript

        gr.Markdown(
            """Made with love by Ilaria 💖 | Support me on [Ko-Fi](https://ko-fi.com/ilariaowo) | Using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

        """
        )

    app.launch()


if __name__ == "__main__":
    main()
