# Nessecary Imports
from pathlib import Path
import best_image_post_processing.target_object as target_object
import best_image_post_processing.mask_generation as mask_generation
import best_image_post_processing.mask_image as mask_image
import best_image_post_processing.get_object_img_masks as get_object_img_masks
import best_image_post_processing.combine_two_mask as combine_two_mask
import best_image_post_processing.nano_banana as nano_banana
import best_image_post_processing.white_background as white_background

from dotenv import load_dotenv


load_dotenv()



# Pipeline method
def image_post_processing(image_path, 
                          object_detection_json_name, 
                          object_detection_debug, 
                          product_url, 
                          target_object_json, 
                          output_directory, 
                          grounded_sam_segmented_image, 
                          yolo_masks_directory, 
                          image_padding, 
                          grounded_sam_output_directory, 
                          nana_banana_generation_mask, 
                          nano_banana_image_path):
    
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

    # Target object call: Target_object.py
    target_object.run_product_cropping_pipeline(
        image_path=image_path,
        product_url=product_url,
        out_dir=output_directory,
    )

    # Grounded SAM call: mask_generation.py
    runner = mask_generation.MaskGenerationRunner()
    runner.run(
        image_path = image_path,
        product_url = product_url,
    )

    # Masked image call: mask_image.py
    """
    Get the third image from the output folder
    """
    first_file = next(Path("best_image_post_processing/crops").glob("*"), None)
    second_file = sorted(p for p in Path("best_image_post_processing/output").iterdir() if p.is_file())[2]
    third_file = "best_image_post_processing/output/kept_with_white_bg.png"

    print("First_file: ", first_file)
    print("Second_file: ", second_file)
    print("Third file: ", third_file)

    mask_image.main(
        original_image = first_file,
        mask_image= second_file,
        out_image=third_file,
    )

    # Yolo Mask call: get_object_img_masks.py
    get_object_img_masks.crop_white_masks_from_merged(
        image_path=image_path,
        merged_json_path="best_image_post_processing/best_image_yolo11_o365_seg.json",
        out_dir=yolo_masks_directory,
        pad=image_padding,
    )

    # Combined Mask call: combine_two_mask.py
    first_image_mask = next(iter(sorted(Path("best_image_post_processing/object_white_masks").glob("*"))), None)
    output_path= "best_image_post_processing/mask_non_overlapping.png"

    print("First_image_mask: ", first_image_mask)
    print("Second_image_mask: ", second_file)
    print("output path: ", output_path)
    combine_two_mask.xor_two_masks(
        mask_a_path=first_image_mask,
        mask_b_path=second_file,
        out_path=output_path,
    )



    # Nano banana call: Nano_banana.py
    nano_banana.run_full_inpaint_with_manual_mask(
        original_image_path="best_image_post_processing/output/kept_with_white_bg.png",
        mask_image_path=nana_banana_generation_mask,
        product_url=product_url,
    )


    # White background call: white_background.py
    white_background.nano_banana_black_bg_to_white(nano_banana_image_path)



# Main class
if __name__ == "__main__":

    # Object detection variables: object_detection.py
    image_path = "best_image.jpg"
    object_detection_json_name = "best_image_yolo11_o365.json"
    object_detection_debug = "best_image_yolo11_o365.jpg"

    # Target object variables: Target_object.py
    product_url = (
        "https://www.amazon.com/MODNEST-Modular-Sectional-Boneless-Assembly/dp/"
        "B0FCYC86TT/ref=sr_1_5_sspa?crid=3RIV4C6CTWYQA&dib=eyJ2IjoiMSJ9."
        "mCR5SjVr3IuoofQI97UxVmePO3nKmQbrqGkH6q7BhCWvXZfA2gaJDgyWsF100Jp3IznRSrEL8WKrwF2Xtlr-Q6YdVwugk_h-"
        "vhluo-EvhxJBqShb2gctTmjV71AXRyHOoPE6xC5K1iS8ITO3gdrhSf93HanYw7yk5iuIDU0gQFvfQLiPHo05ZX5PuYYk5As943eAeCxhe_d7i07UPtixVaCT_4yDty6lWukpMHvQmtssdgjG_"
        "zKvPqz8uqzZ5oAjIljfU3T1fj2lJKgKnqjlWxjA504G0RVwfRlQuUKr5bM.bAdATDJIW3OxMs6M16JiHRG9zAvYg7_B-SoRvPV4i5s&dib_tag=se&keywords=couch&qid=1762565138&"
        "sprefix=cou%2Caps%2C264&sr=8-5-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"
    )
    target_object_json = "best_image_yolo11_o365_merged_alias.json"
    output_directory = "crops"

    # Grounded SAM variables: mask_generation.py


    # Masked image variables: mask_image.py
    grounded_sam_segmented_image = "output/kept_with_white_bg.png"

    # Yolo Mask variables: get_object_img_masks.py
    yolo_masks_directory = "object_white_masks"
    image_padding = 0

    # Combined Mask variable: combine_two_mask.py
    grounded_sam_output_directory = "output"
    nana_banana_generation_mask = "mask_non_overlapping.png"

    # Nano banana variables: Nano_banana.py
    nano_banana_image_path = "uploads/inpainted_manual_mask.png"

    image_post_processing(
        image_path=image_path, 
        object_detection_json_name=object_detection_json_name, 
        object_detection_debug=object_detection_debug, 
        product_url=product_url,
        target_object_json=target_object_json, 
        output_directory=output_directory,
        grounded_sam_segmented_image=grounded_sam_segmented_image,
        yolo_masks_directory=yolo_masks_directory,
        image_padding=image_padding,
        grounded_sam_output_directory=grounded_sam_output_directory,
        nana_banana_generation_mask=nana_banana_generation_mask,
        nano_banana_image_path=nano_banana_image_path
    )



