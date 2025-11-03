#!/usr/bin/env python3
# pip install selenium
# (Optional) For better evasion, consider undetected-chromedriver:
#   pip install undetected-chromedriver

import time, random, sys
from typing import Optional, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, JavascriptException

WAYFAIR_URL = (
    "https://www.wayfair.com/furniture/pdp/latitude-run-cenie-modern-upholstered"
    "-arc-shaped-3d-knit-fabric-sofa-no-assembly-required3-seat-w115334476.html?"
    "piid=224002162"
)

PRESS_HOLD_TEXT_XPATH = (
    # Any element whose visible text includes both "press" and "hold"
    "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'press')"
    " and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'hold')]"
)

PRESS_HOLD_ROLE_BUTTON = (
    "//*[@role='button' and (contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'press')"
    " and contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'hold'))]"
)

HUMAN_IFRAME_GUESSES = [
    "human", "perimeterx", "px", "captcha", "arkose", "challenge", "hcaptcha"
]

def init_chrome(headless=False) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,1000")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=opts)
    # Reduce obvious automation fingerprints a bit
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    })
    return driver

def wait_dom_ready(driver, timeout=20):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
    )

def in_iframe_find_press_hold(driver, timeout=5) -> Tuple[Optional[object], Optional[int]]:
    """Search main page first, then all iframes, return (element, frame_index)."""
    wait = WebDriverWait(driver, timeout)
    # main DOM
    try:
        el = wait.until(EC.presence_of_element_located((By.XPATH, f"({PRESS_HOLD_ROLE_BUTTON}) | ({PRESS_HOLD_TEXT_XPATH})")))
        return el, None
    except TimeoutException:
        pass

    frames = driver.find_elements(By.CSS_SELECTOR, "iframe, frame")
    for idx, fr in enumerate(frames):
        # prioritize likely HUMAN/PerimeterX frames
        src = (fr.get_attribute("src") or "").lower()
        title = (fr.get_attribute("title") or "").lower()
        hint = any(h in src or h in title for h in HUMAN_IFRAME_GUESSES)
        try_order = [fr] if hint else []
        # If no hint, we’ll check all anyway
    # Try hinted first, then all
    order = sorted(range(len(frames)),
                   key=lambda i: 0 if any(h in ((frames[i].get_attribute("src") or "").lower() + " " + (frames[i].get_attribute("title") or "").lower()) for h in HUMAN_IFRAME_GUESSES) else 1)

    for i in order:
        try:
            driver.switch_to.frame(frames[i])
            try:
                el = wait.until(EC.presence_of_element_located((By.XPATH, f"({PRESS_HOLD_ROLE_BUTTON}) | ({PRESS_HOLD_TEXT_XPATH})")))
                return el, i
            except TimeoutException:
                pass
        finally:
            driver.switch_to.default_content()
    return None, None

def mouse_press_hold(driver, el, hold_seconds=4.0):
    actions = ActionChains(driver)
    actions.move_to_element(el).perform()
    time.sleep(0.15)
    actions.click_and_hold(el).perform()

    # subtle human-like jitter while holding
    t_end = time.time() + hold_seconds
    while time.time() < t_end:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        ActionChains(driver).move_by_offset(dx, dy).perform()
        time.sleep(0.15 + random.random() * 0.15)

    ActionChains(driver).release(el).perform()

def js_pointer_hold(driver, el, hold_seconds=4.0):
    js = r"""
const el = arguments[0];
const holdMs = Math.max(0, Math.floor(arguments[1]*1000));
el.scrollIntoView({block:'center', inline:'center'});
function fire(type, opts={}) {
  const r = el.getBoundingClientRect();
  const evt = new PointerEvent(type, Object.assign({
    bubbles: true, cancelable: true, composed: true,
    pointerId: 1, pointerType: 'mouse', isPrimary: true,
    clientX: (r.left + r.right)/2, clientY: (r.top + r.bottom)/2,
    buttons: 1
  }, opts));
  el.dispatchEvent(evt);
}
function mouse(type){ el.dispatchEvent(new MouseEvent(type,{bubbles:true,cancelable:true,buttons:1})); }
function touch(type){
  try{
    const r=el.getBoundingClientRect();
    const t=new Touch({identifier:1,target:el,clientX:(r.left+r.right)/2,clientY:(r.top+r.bottom)/2});
    const e=new TouchEvent(type,{bubbles:true,cancelable:true,touches:[t],targetTouches:[t],changedTouches:[t]});
    el.dispatchEvent(e);
  }catch(_){}
}
fire('pointerover'); fire('pointerenter'); fire('pointerdown');
mouse('mouseover'); mouse('mouseenter'); mouse('mousedown'); touch('touchstart');

return new Promise(res=>{
  setTimeout(()=>{
    fire('pointerup',{buttons:0});
    mouse('mouseup'); mouse('click'); touch('touchend');
    res(true);
  }, holdMs);
});
"""
    driver.execute_script(js, el, hold_seconds)
    time.sleep(hold_seconds + 0.25)

def solve_press_hold(driver, max_attempts=3, hold_seconds=4.0) -> bool:
    """Try to pass the HUMAN ‘Press & Hold’ up to max_attempts times."""
    for attempt in range(1, max_attempts+1):
        el, frame_index = in_iframe_find_press_hold(driver, timeout=8)
        if not el:
            # no challenge element right now
            return True

        # switch into frame if needed
        if frame_index is not None:
            frames = driver.find_elements(By.CSS_SELECTOR, "iframe, frame")
            try:
                driver.switch_to.frame(frames[frame_index])
            except Exception:
                driver.switch_to.default_content()

        # try mouse press-and-hold, then JS fallback
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable(el))
        except TimeoutException:
            pass

        try:
            mouse_press_hold(driver, el, hold_seconds)
        except WebDriverException:
            try:
                js_pointer_hold(driver, el, hold_seconds)
            except JavascriptException:
                pass

        # back to main content and check if challenge disappeared
        driver.switch_to.default_content()
        time.sleep(1.0)
        still_there, _ = in_iframe_find_press_hold(driver, timeout=3)
        if not still_there:
            return True

        # small backoff before retry
        time.sleep(1.5 + attempt * 0.75)

    return False

def main(url=WAYFAIR_URL, headless=False):
    driver = init_chrome(headless=headless)
    try:
        driver.get(url)
        wait_dom_ready(driver, 25)

        # If challenge is present, try to solve
        if not solve_press_hold(driver, max_attempts=3, hold_seconds=4.0):
            print("Could not pass the Press & Hold gate after retries.")
            return

        # At this point we should be on the product page
        print("Page title:", driver.title)
        print("URL:", driver.current_url)

    finally:
        driver.quit()

if __name__ == "__main__":
    # Run headful first so you can watch what happens
    main(headless=False)
