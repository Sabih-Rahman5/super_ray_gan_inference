from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import time
import cv2

# Read the image
image = cv2.imread('inputs/3.jpg')




# Get the dimensions (height, width, and number of channels)
height, width, channels = image.shape


print(f"Height: {height} pixels")
print(f"Width: {width} pixels")
print(f"Number of Channels: {channels}")


print(f"Image Shape: {image.shape}")


path = 'superray_fracture_20000.pth'

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)


upsampler = RealESRGANer(
        scale=4,
        model_path= 'weights/' + str(path),
        dni_weight= None,
        model=model,
        tile= 0,
        tile_pad= 10,
        pre_pad= 0,
        half= False,
        gpu_id= None)

start = time.time()
output, _ = upsampler.enhance(image, outscale=4)
end = time.time()


print("runtime: ", (end-start), "seconds")


cv2.imwrite("result_image_normal.jpg", output)













