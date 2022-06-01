# Image Eraser

This tool allows removing objects from images. It takes two images, the source image and a mask. The mask contains the region
which should be inpainted.

> Caution: This tool is resource intensive! It can take hours of magnitude to finish based on the input image and mask.

# Usage

First install all the requirements with:

```
pip install -r requirements.txt
```

The pipeline can then be run with the `main.py` script:

```
python main.py
```

## Results

Below are some results obtained from using this tool.

<p align="center">
	<img src="images/cat/input_1.jpg" alt="input_1" style="width:350px;"/>
	<img src="images/cat/result_1.png" alt="result_1" style="width:350px;"/>
	<img src="images/cat/mask_1.jpg" alt="mask_1" style="width:350px;"/>
</p>

<p align="center">
	<img src="images/bird/input_1.jpg" alt="input_1" style="width:350px;"/>
	<img src="images/bird/result_1.png" alt="result_1" style="width:350px;"/>
	<img src="images/bird/mask_1.jpg" alt="mask_1" style="width:350px;"/>
</p>
