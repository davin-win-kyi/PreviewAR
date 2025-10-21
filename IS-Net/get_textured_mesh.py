import argparse, os, time, shutil, glob, pathlib
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from Inference import dis

from company_agent import process_url

from blender_scaling import get_dimensions_via_gpt, scale_glb_in_blender

import json

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
    # out_path = process_url("https://www.amazon.com/MAXYOYO-Boneless-Backrest-Upholstered-Armrests/dp/B0FLDP6QW9/ref=sr_1_26?crid=368RDRS52OHRH&dib=eyJ2IjoiMSJ9.UxtiqFhNvt8FIthZUZvN2Mo8HtpaxZuo7k1psGyfivTAhtkOFeqDYiGIPIAiZ93fZ6uX58lMtgb8aw4ESt4G0uOo0rabEETrSqqk0b_VlwWT_WZ_ACU3yqNeeNugINEEGLhgPInP1DyhZVCQuNvNjFl64AzoJndTjRlDgA7QH9uRGUmPjpUe3BxAKFWBnL__CATDEfoLxcDtmUx2Jf7ud3PH3quw1c2LNNE84eaGkBv1dLM6HhZpxpbu39hzSeRHBVjQ_RBKsgMDqn9fIRM2yc_FvIIp-gASDXlk2dM3b_I.IiO8pW-3VmMUsh2dFdwAV3eARcZO66WEYcrSXaEb5KA&dib_tag=se&keywords=couch&qid=1761027647&s=home-garden&sprefix=couch%2Cgarden%2C214&sr=1-26", "test.json")

    # now process the json with output information
    # dis(out_path)
    dis("test.json")



    # setting arguments for later use
    ap = argparse.ArgumentParser(description="Drive Hunyuan3D Gradio with Selenium and fetch textured mesh.")
    ap.add_argument("--url", default="http://localhost:1080//", help="Gradio URL")
    ap.add_argument("--image", default="best_image.png", help="Path to input image")
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
        out_path = pathlib.Path(args.out)
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
