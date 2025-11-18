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
        https://www.amazon.com/MODNEST-Modular-Sectional-Boneless-Assembly/dp/B0FCYC86TT/ref=sr_1_5_sspa?crid=3RIV4C6CTWYQA&dib=eyJ2IjoiMSJ9.mCR5SjVr3IuoofQI97UxVmePO3nKmQbrqGkH6q7BhCWvXZfA2gaJDgyWsF100Jp3IznRSrEL8WKrwF2Xtlr-Q6YdVwugk_h-vhluo-EvhxJBqShb2gctTmjV71AXRyHOoPE6xC5K1iS8ITO3gdrhSf93HanYw7yk5iuIDU0gQFvfQLiPHo05ZX5PuYYk5As943eAeCxhe_d7i07UPtixVaCT_4yDty6lWukpMHvQmtssdgjG_zKvPqz8uqzZ5oAjIljfU3T1fj2lJKgKnqjlWxjA504G0RVwfRlQuUKr5bM.bAdATDJIW3OxMs6M16JiHRG9zAvYg7_B-SoRvPV4i5s&dib_tag=se&keywords=couch&qid=1762565138&sprefix=cou%2Caps%2C264&sr=8-5-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1
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
    
    post_processed_image = "uploads/post_processing_image.png"

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
