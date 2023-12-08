![Ilaria AI Suite](./ilariaaisuite.png)
***
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-s?labelColor=YELLOW&color=FFEA00)](https://huggingface.co/spaces/TheStinger/Ilaria_Upscaler) [![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-Duplication-s?labelColor=YELLOW&color=FFEA00)](https://huggingface.co/spaces/TheStinger/Ilaria_Upscaler?duplicate=true) [![Static Badge](https://img.shields.io/badge/GitHub-Source%20Code-s?logo=GitHub)](https://github.com/TheStingerX/Ilaria-Upscaler) [![Static Badge](https://img.shields.io/badge/AI%20Hub-Discord%20Server-s?logo=Discord&color=%09%237289da)](https://discord.gg/aihub) [![Static Badge](https://img.shields.io/badge/Ko--Fi-s?logo=Ko-Fi&label=Support%20me%20on&labelColor=434b57&color=FF5E5B)](https://ko-fi.com/ilariaowo)
***
<p align="center">
  <h1>Ilaria Upscaler 💖</h1>
</p>

🎉 Welcome to Ilaria Upscaler! 🎉  
  
This project leverages various libraries and modules to create a Graphical User Interface (GUI) for upscaling images.  
It's primarily designed for use with HuggingFace Spaces. 🤗   

Ilaria Upscaler is part of the Ilaria AI Suite wich includes various easy and powerful tools. 💖

## 📦 Installation 📦

To use this project, clone the original Space on Hugging Face.  
Make sure you restart it time to time to keep up with the new updates.

## 🖥️ Usage 🖥️

Once the dependencies are installed automatically, Hugging Face will use app.py to start the user interface.  
From there, you can utilize the various features of the project.

## 🌟 Features 🌟

Ilaria Upscaler offers a range of features, including:

- 🖼️ **Image Upscaling and Restoration**:  
Ilaria Upscaler uses the RealESRGANer class from the realesrgan module to upscale and restore images.
It supports multiple models, each with different scales and architectures, providing flexibility in the upscaling process.

- 🎛️ **Adjustable Parameters**:  
The realesrgan function allows users to adjust various parameters such as the model name, denoise strength, face enhancement, and output scale.
This provides a high level of customization to fit different use cases.

- 🖥️ **User-Friendly Web Interface**:  
Ilaria Upscaler The code uses the gradio module to create an intuitive web interface.
Users can easily upload an image, and the application will return the upscaled and restored image.

- 📂 **Efficient File Handling**:  
The code uses the os and basicsr.utils.download_util modules for efficient file handling operations.
It checks if the model file exists in the local directory, and if not, it downloads the file from the specified URL.

- 👩 **Face Enhancement Option**:  
If the face_enhance parameter is set to True, the code uses the GFPGANer class from the gfpgan module to enhance faces in the image.  
This is particularly useful for images with faces that need to be upscaled and enhanced separately from the rest of the image.

## 🙏 Credits 🙏

- **Squishy** - For helping me with the ideas

## 🤝 Contributing 🤝

Interested in contributing to this project? Ilaria is always looking for collaborators.  
Feel free to open a pull request on Hugging Face.

## 📄 License 📄

This project is released under the `INCU` license.  
For more details, please check the license file.  
For further questions feel free to contact Ilaria.
