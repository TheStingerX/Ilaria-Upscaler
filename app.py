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
    """Real-ESRGAN function to restore (and upscale) images."""
    if not img:
        return

    # Define model parameters
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

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

    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    cv_img = numpy.array(img)
    img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        extension = 'png' if img_mode == 'RGBA' else 'jpg'

        out_filename = f"output_{rnd_string(8)}.{extension}"
        cv2.imwrite(out_filename, output)
        global last_file
        last_file = out_filename

        output_img = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA) if img_mode == "RGBA" else output
        return out_filename, image_properties(output_img)

def rnd_string(x):
    characters = "abcdefghijklmnopqrstuvwxyz_0123456789"
    return "".join((random.choice(characters)) for i in range(x))

def reset():
    global last_file
    if last_file:
        print(f"Deleting {last_file} ...")
        os.remove(last_file)
        last_file = None
    return gr.update(value=None), gr.update(value=None), gr.update(value=None)

def has_transparency(img):
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
    if img is None:  # Explicitly check for None
        return "No image data available."

    if isinstance(img, numpy.ndarray):  # Handle NumPy array case
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        img_mode = "RGBA" if channels == 4 else "RGB" if channels == 3 else "Grayscale"
        return f"Resolution: Width: {width}, Height: {height}  |  Color Mode: {img_mode}"
    
    if hasattr(img, "info") and hasattr(img, "mode") and hasattr(img, "size"):  # Handle PIL images
        if has_transparency(img):
            img_mode = "RGBA"
        else:
            img_mode = "RGB"
        return f"Resolution: Width: {img.size[0]}, Height: {img.size[1]}  |  Color Mode: {img_mode}"
    
    return "Unsupported image format."

def main():
    with gr.Blocks(theme=gr.themes.Default(primary_hue="pink", secondary_hue="rose"), title="Ilaria Upscaler 💖") as app:

        gr.Markdown(
            """# <div align="center"> Ilaria Upscaler 💖 </div>  
        """
        )
        with gr.Accordion("Upscaling option"):
            with gr.Row():
                model_name = gr.Dropdown(label="Model", 
                                        choices=["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x2plus", "realesr-general-x4v3"], 
                                        value="RealESRGAN_x4plus")
                denoise_strength = gr.Slider(label="Denoise Strength", minimum=0, maximum=1, step=0.1, value=0.5)
                outscale = gr.Slider(label="Resolution Upscale", minimum=1, maximum=6, step=1, value=4)
                face_enhance = gr.Checkbox(label="Face Enhancement")

        with gr.Row():
            with gr.Group():
                input_image = gr.Image(label="Input Image", type="pil")
                input_properties = gr.Textbox(label="Input Image Properties", interactive=False)

            with gr.Group():
                output_image = gr.Image(label="Output Image")
                output_properties = gr.Textbox(label="Output Image Properties", interactive=False)

        with gr.Row():
            reset_btn = gr.Button("Reset")
            upscale_btn = gr.Button("Upscale")

        input_image.change(fn=image_properties, inputs=input_image, outputs=input_properties)
        upscale_btn.click(fn=realesrgan, 
                        inputs=[input_image, model_name, denoise_strength, face_enhance, outscale], 
                        outputs=[output_image, output_properties])
        reset_btn.click(fn=reset, inputs=[], outputs=[input_image, output_image, input_properties])

        gr.Markdown(
            """Made with love by Ilaria 💖 | Support me on [Ko-Fi](https://ko-fi.com/ilariaowo) | Using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
        """
        )

    app.launch()

if __name__ == "__main__":
    main()
