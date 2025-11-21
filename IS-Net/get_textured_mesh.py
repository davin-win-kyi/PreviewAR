import argparse, os, time, shutil, glob, pathlib
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from best_image_selector import get_best_image_url

from Inference import dis

from company_agent import process_url

from blender_scaling import get_dimensions_via_gpt, scale_glb_in_blender

import json


from best_image_post_processing import image_post_processing


import os
import shutil
from pathlib import Path

def clean_best_image_artifacts(base_dir: str = "."):
    """
    Deletes specific folders and files under best_image_post_processing,
    then recreates the folders (empty).

    Folders removed & recreated:
      - best_image_post_processing/crops
      - best_image_post_processing/object_white_masks
      - best_image_post_processing/output
      - best_image_post_processing/uploads

    Files removed (not recreated):
      - best_image_post_processing/best_image_yolo11_o365_seg.jpg
      - best_image_post_processing/best_image_yolo11_o365_seg.json
    """
    base_path = Path(base_dir) / "best_image_post_processing"

    # Folders to delete & later recreate
    folders = [
        "crops",
        "object_white_masks",
        "output",
        "uploads",
    ]

    # Files to delete
    files = [
        "best_image_yolo11_o365_seg.jpg",
        "best_image_yolo11_o365_seg.json",
    ]

    # Delete folders
    for folder in folders:
        folder_path = base_path / folder
        if folder_path.exists() and folder_path.is_dir():
            print(f"Deleting folder: {folder_path}")
            shutil.rmtree(folder_path)
        else:
            print(f"Folder not found (skipping): {folder_path}")

    # Delete files
    for file_name in files:
        file_path = base_path / file_name
        if file_path.exists() and file_path.is_file():
            print(f"Deleting file: {file_path}")
            file_path.unlink()
        else:
            print(f"File not found (skipping): {file_path}")

    # Recreate folders (empty)
    for folder in folders:
        folder_path = base_path / folder
        print(f"Recreating folder: {folder_path}")
        folder_path.mkdir(parents=True, exist_ok=True)



def newest_glb_from_cache(cache_dir: str, started_after: float):
    """
    Return path to newest textured_mesh.glb created
    """
    if not os.path.isdir(cache_dir):
        return None
    candidates = []
    for d in glob.glob(os.path.join(cache_dir, "*")):
        if not os.path.isdir(d):
            continue
        try:
            mtime = os.path.getmtime(d)
        except FileNotFoundError:
            continue
        if mtime < started_after:
            continue
        glb = os.path.join(d, "textured_mesh.glb")
        if os.path.isfile(glb):
            candidates.append((mtime, glb))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def main():

    """
    In the beginning make sure to, clear all the folders
    so that none of the previous crops, object_white_masks, output, uploads are present
    - also make sure to delete the seg jsons from the previous runs
    """
    clean_best_image_artifacts()

    # handling the amazon case right now
    # out_path = process_url("https://www.amazon.com/Sectional-Minimalist-Upholstered-Couch%EF%BC%8CNo-Assembly/dp/B0DMSPJ97J/ref=sxin_16_pa_sp_search_thematic_sspa?content-id=amzn1.sym.a1bc2dac-8d07-44d1-9477-59bc11451909%3Aamzn1.sym.a1bc2dac-8d07-44d1-9477-59bc11451909&crid=1X7V4GO2K8PE9&cv_ct_cx=couch&keywords=couch&pd_rd_i=B0DMSPJ97J&pd_rd_r=00c418e9-74a0-40d2-882c-295d127e6cef&pd_rd_w=e9VtT&pd_rd_wg=lfOx6&pf_rd_p=a1bc2dac-8d07-44d1-9477-59bc11451909&pf_rd_r=6VAC3N6XWY0GH5P85MSB&qid=1761351451&s=home-garden&sbo=RZvfv%2F%2FHxDF%2BO5021pAnSA%3D%3D&sprefix=couch%2Cgarden%2C391&sr=1-2-9428117c-b940-4daa-97e9-ad363ada7940-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9zZWFyY2hfdGhlbWF0aWM&th=1", "test.json")
    url = (
        # "https://www.amazon.com/Sectional-Minimalist-Upholstered-Couch%EF%BC%8CNo-Assembly/dp/B0DMSNCX14/ref=sr_1_1_sspa"
        # "?crid=3Q0OC9EF9BOT2"
        # "&dib=eyJ2IjoiMSJ9.Uwy_-hTxn36mxYatk6YVYoZzfr9ccOrbiBYTzPXlkhX20Xljw7XFV30e8JTA_UIVAcnSUfDH6SdliqACjdbtTxjItAW9S6wE3RCmOValBQUGnzlCgRtfgk4fa-PzKL8th62Cz6rAe5mruSurnxNcQ4vdjN_j0FIIIrxNqwaXdeeWa4zdYX7h608_MdeH7Xej50FqMcTQb_HicnZzBSAQVlt295PrnBXwNELEt5T-1MFOtNIs_4fB2vVpJb6X5ZdbREdGQxJexPzxwM9GK0X86-1R1IhzscV8fquOFk9dwMk.SxonPO9dTDRt6Xrhq1MNRk2KVFfS9rSsWmQ8r_nFdNE"
        # "&dib_tag=se"
        # "&keywords=couch"
        # "&qid=1762054233"
        # "&sprefix=couch%2Caps%2C195"
        # "&sr=8-1-spons"
        # "&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY"
        # "&th=1"

        """
        https://www.amazon.com/Vesgantti-Cloud-Sectional-Chenille-Left-Facing/dp/B0FKGSHC74/ref=sr_1_2_sspa?dib=eyJ2IjoiMSJ9.1zBsCSqJxO1X6i2gfj5yhJjnxWCqmZPx5tTCyRXSH8npv2Q_t0g_kMjhbY_e1soTN1Fj-sRK8Ugkx3Krbm8B-uHVbywxrdaAgf4lJGADLa0GLj5LNK_ifGfd6LlzSOFfXLPe_Q7DHB_NDTgp9S3Ql7lYt-V2HL9YFvB-8B9YByPIWBOVjsiNC3O9C9-b4NUawxb-mbg-oQhWVZe7vTs84GiqYr5-UDjXgFWgqz41glJV6OWcmEMhO8wEPNdan2UaBbiP-p67qjD6I8pZqScDZL2NB04x3qevUIu5o2aLgj4.yZVLUFl632Qpdv2phmDAWmrJCjIdX7Dhq1gNRG_P4Q0&dib_tag=se&keywords=couch&qid=1763603235&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1
        """
    )
        
    get_best_image_url(url)

    # now process the json with output information
    dis()
    # dis("test.json")

    # input the image as well as the test.png file name into the method call image_post_processing.py
    # and call the method 
    # image_to_process = "test.png"

    # here is the call to the post_processing_pipeline
    """
    TODO: put the variables you had from the main in image_post_processing.py here
    and also put in the right parameters into image_post_processing
    """
        # Object detection variables: object_detection.py
    image_path = "best_image.png"
    object_detection_json_name = "best_image_post_processing/best_image_yolo11_o365.json"
    object_detection_debug = "best_image_post_processing/best_image_yolo11_o365.jpg"

    # Target object variables: Target_object.py
    product_url = (
        "https://www.amazon.com/Vesgantti-Cloud-Sectional-Corduroy-Right-Facing/dp/B0FNRNFH23/ref=sr_1_6?crid=1RGQC8GP21GVE&dib=eyJ2IjoiMSJ9.6yhPLIAaV77WfAePh8vDx5zfgTJUi1jvY7S1GR2vUass_EUWjVrPYbFkRb172hZqQ_B5toEm-PCe3mJBq1utg0nn6ZKJ0sj8rutPEFhKt0dH2uL4QCKN9GRHu-67LTHt8-DS-PJVapyhUh1adGW0GaslDNlgQNXs-bjwDL0xM4cDb4WFVMsY5uAMHagfau2XkuRquc5PoYEyWSc5RYYunPxVo8M6CvvaUmSsrFTKt3c3rqD4Z5iB8YAvjEwTDy62b-MX3BZ-cEK085tUPU4AQrh2KohRHte39FJrbIIYN1s.Stad0IZR62UHGKGGwDIZap28tlvgu-g7tr6Po0n216g&dib_tag=se&keywords=couch&qid=1763540421&sprefix=couc%2Caps%2C249&sr=8-6&ufe=app_do%3Aamzn1.fos.5998aa40-ec6f-4947-a68f-cd087fee0848&th=1"
    )
    target_object_json = "best_image_post_processing/previewar_test#1_yolo11_o365_merged_alias.json"
    output_directory = "best_image_post_processing/crops"

    # Grounded SAM variables: mask_generation.py


    # Masked image variables: mask_image.py
    grounded_sam_segmented_image = "best_image_post_processing/output/kept_with_white_bg.png"

    # Yolo Mask variables: get_object_img_masks.py
    yolo_masks_directory = "best_image_post_processing/object_white_masks"
    image_padding = 0

    # Combined Mask variable: combine_two_mask.py
    grounded_sam_output_directory = "best_image_post_processing/output"
    nana_banana_generation_mask = "best_image_post_processing/mask_non_overlapping.png"

    # Nano banana variables: Nano_banana.py
    nano_banana_image_path = "best_image_post_processing/uploads/inpainted_manual_mask.png"

    image_post_processing.image_post_processing(
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
    
    post_processed_image = "best_image_post_processing/uploads/post_processing_image.png"

    # setting arguments for later use
    ap = argparse.ArgumentParser(description="Drive Hunyuan3D Gradio with Selenium and fetch textured mesh.")
    ap.add_argument("--url", default="http://localhost:1080//", help="Gradio URL")
    # ap.add_argument("--image", default="test.png", help="Path to input image")
    ap.add_argument("--image", default=post_processed_image, help="Path to input image")
    ap.add_argument("--cache-dir", default=r"C:\Users\davin\OneDrive\Documents\HY3D2\Hunyuan3D2_WinPortable_cu126\Hunyuan3D2_WinPortable\Hunyuan3D-2-vanilla\gradio_cache",
                    help="Path to Hunyuan3D gradio_cache directory")
    ap.add_argument("--timeout", type=int, default=120, help="Max seconds to wait for textured mesh")
    ap.add_argument("--out", default="hy3d_output/model.glb", help="Where to copy the GLB")
    ap.add_argument("--headless", action="store_true", help="Run Chrome headless")
    args = ap.parse_args()

    # Getting image path, enter this into terminal: python get_textured_mesh.py --image <image_path>
    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    # Running Selenium for user
    options = webdriver.ChromeOptions()
    if args.headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")

    # load the webpage, headed to see the changes
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(args.url)
        wait = WebDriverWait(driver, 30)

        # input the image file
        file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        file_input.send_keys(image_path)

        # click the generate textured mesh button 
        try:
            gen_tex_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Gen Textured Shape']"))
            )
        except Exception:
            gen_tex_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Gen Textured Shape')]"))
            )
        t_click = time.time()
        gen_tex_button.click()

        # checking for the new glb file to show up
        deadline = time.time() + args.timeout
        found_glb = None
        last_note = 0
        while time.time() < deadline:
            glb = newest_glb_from_cache(args.cache_dir, started_after=t_click)
            if glb:
                found_glb = glb
                break
            # check to see if a new file is there 
            # checking every 5 seconds
            if time.time() - last_note > 5:
                print("Waiting for textured_mesh.glb to appear in cache...")
                last_note = time.time()
            time.sleep(1.5)

        if not found_glb:
            raise SystemExit(
                f"Timed out after {args.timeout}s: no textured_mesh.glb created under\n  {args.cache_dir}"
            )

        # saving the glb file
        out_path = pathlib.Path("model.glb")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found_glb, out_path)

    finally:
        driver.quit()


    # here get the scaled version of the object
    test_json = json.load(open("test.json", "r", encoding="utf-8"))
    dimensions = get_dimensions_via_gpt(test_json)
    print("Dimensions: ", dimensions)

    # this needs to be done in blender or in Unity
    # scale_glb_in_blender("C:\\Users\\davin\\Hunyuan3D-2-WinPortable\\hy3d_output\\model.glb", dimensions["length_m"], dimensions["width_m"], dimensions["height_m"], "model_scale.glb")
    


if __name__ == "__main__":
    main()
