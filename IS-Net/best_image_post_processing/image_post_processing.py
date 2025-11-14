# Nessecary Imports
import object_detection
import target_object
import mask_generation
import mask_image
import get_object_img_masks
import combine_two_mask
import nano_banana
import white_background



# Pipeline method
def image_post_processing():
    """
    Overall pipeline structure: 
    1. object_detection.py
        - Get the object detections which will be bounding boxes 
          and segmentations of the object 

    2. target_object.py 
        - Get the class name for the object of interest and get
          the bounding_boxes and masks for the target object

    3. mask_generation.py 
        - Get the grounded sam mask of the image

    4. mask_image.py 
        - Get the segmented image using the grounded sam mask 

    5. get_object_img_masks.py
        - Get the Yolo masks of the target object

    6. combined_two_mask.py 
        - Get the mask where you have the parts of the original 
          image that you want to modify through stable diffusion

    7. Nano_banana.py 
        - here you will pass in the combined mask along with
          a prompt to generate the final image

    8. white_background.py
        - here you will set the background of the image to 
          be white
    
    """
    # TODO:

    # Object detection call: object_detection.py


    # Target object call: Target_object.py


    # Grounded SAM call: mask_generation.py


    # Masked image call: mask_image.py
    """
    Get the third image from the output folder
    """


    # Yolo Mask call: get_object_img_masks.py


    # Combined Mask call: combine_two_mask.py


    # Nano banana call: Nano_banana.py


    # White background call: white_background.py



# Main class
if __name__ == "__main__":

    # Object detection variables: object_detection.py
    image_path = "previewar_test#1.jpg"
    object_detection_json_name = "previewar_test#1_yolo11_o365.json"
    object_detection_debug = "previewar_test#1_yolo11_o365.jpg"

    # Target object variables: Target_object.py
    product_url = (
        "https://www.amazon.com/MODNEST-Modular-Sectional-Boneless-Assembly/dp/"
        "B0FCYC86TT/ref=sr_1_5_sspa?crid=3RIV4C6CTWYQA&dib=eyJ2IjoiMSJ9."
        "mCR5SjVr3IuoofQI97UxVmePO3nKmQbrqGkH6q7BhCWvXZfA2gaJDgyWsF100Jp3IznRSrEL8WKrwF2Xtlr-Q6YdVwugk_h-"
        "vhluo-EvhxJBqShb2gctTmjV71AXRyHOoPE6xC5K1iS8ITO3gdrhSf93HanYw7yk5iuIDU0gQFvfQLiPHo05ZX5PuYYk5As943eAeCxhe_d7i07UPtixVaCT_4yDty6lWukpMHvQmtssdgjG_"
        "zKvPqz8uqzZ5oAjIljfU3T1fj2lJKgKnqjlWxjA504G0RVwfRlQuUKr5bM.bAdATDJIW3OxMs6M16JiHRG9zAvYg7_B-SoRvPV4i5s&dib_tag=se&keywords=couch&qid=1762565138&"
        "sprefix=cou%2Caps%2C264&sr=8-5-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"
    )
    target_object_json = "previewar_test#1_yolo11_o365_merged_alias.json"
    output_directory = "crops"

    # Grounded SAM variables: mask_generation.py


    # Masked image variables: mask_image.py
    grounded_sam_segmented_image = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"

    # Yolo Mask variables: get_object_img_masks.py
    yolo_masks_directory = "object_white_masks"
    image_padding = 0

    # Combined Mask variable: combine_two_mask.py
    grounded_sam_output_directory = "output"
    nana_banana_generation_mask = "mask_non_overlapping.png"

    # Nano banana variables: Nano_banana.py
    nano_banana_image_path = "/uploads/inpainted_manual_mask.png"



