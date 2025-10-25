import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from web_scraping_embeddings import get_rag_db, query_rag_for_dimensions, query_rag_for_images
from openai import OpenAI
import re
from dotenv import load_dotenv

load_dotenv()

def analyze_product_with_rag(html_content):
    """Use RAG to identify product dimensions and images"""
    print("Loading RAG database...")
    db, embeddings = get_rag_db()
    
    print("Querying for product dimension selectors...")
    query_rag_for_dimensions(db)
    
    print("Querying for product image selectors...")
    query_rag_for_images(db)
    
    # Use OpenAI to analyze the HTML with RAG context
    client = OpenAI()
    
    # Use the entire HTML file with a more efficient approach
    prompt = f"""Analyze this complete AMAZON product page HTML to find product dimensions and main image.

Amazon selectors to look for:
- Dimensions: #productDetails_techSpec_section_1, #productDetails_detailBullets_sections1
- Images: #landingImage, .imgTagWrapper img

HTML (img and dimension-related tags (td or span)):
{html_content}

Instructions:
1. Search through the entire HTML for the Amazon selectors above
2. Extract actual dimension values (not just selectors)
3. Extract the main product image URL (not just selectors)
4. Look for text containing measurements like "inches", "cm", "dimensions"
5. Look for img tags with src containing "media-amazon.com"

Format your response as:
DIMENSIONS: [specific measurements found or "Not found"]
IMAGE_URL: [full image URL or "Not found"]"""
    
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response.choices[0].message.content

def main():
    driver = webdriver.Chrome()

    # driver.get('https://www.amazon.com/dp/B07BQL8L1K?ref=cm_sw_r_cso_cp_apin_dp_GW6WMDZ8H6Q30EDBCSKD&ref=cm_sw_r_cso_cp_apin_dp_GW6WMDZ8H6Q30EDBCSKD&social_share=cm_sw_r_cso_cp_apin_dp_GW6WMDZ8H6Q30EDBCSKD&titleSource=true&th=1')
    # driver.get('https://www.ikea.com/us/en/p/finnala-cover-sleeper-sofa-w-chaise-gunnared-beige-s09575625/?gad_source=1&gad_campaignid=22409704213&gbraid=0AAAAAD27g7wHUXffYnx1K-RqXJqLbJBoN&gclid=CjwKCAjwxrLHBhA2EiwAu9EdMxJ69Nh_OxEXD2IpAA_IfzGzmN_XMo_NVFb3SOqTZcacEbXiB0BTTxoC4AQQAvD_BwE')
    # driver.get('https://www.ebay.com/itm/127072625292?chn=ps&var=428200680998&norover=1&mkevt=1&mkrid=711-166974-028196-7&mkcid=2&mkscid=101&itemid=428200680998_127072625292&targetid=2274951440814&device=c&mktype=pla&googleloc=9033309&poi=&campaignid=22634827372&mkgroupid=184329790790&rlsatarget=pla-2274951440814&abcId=10337614&merchantid=5598018643&geoid=9033309&gad_source=1&gad_campaignid=22634827372&gbraid=0AAAAAD_QDh_bRS0NG5nee2YUrWYc0PmA1&gclid=CjwKCAjwxrLHBhA2EiwAu9EdMzyLAL6mCEXmW18U_upm0WDn8X3zmG5oZcUmkRLAELHtS1iSnpKy9xoCa1gQAvD_BwE')
    driver.get('https://www.wayfair.com/furniture/pdp/ivy-bronx-nathasa-2-piece-no-assembly-required-upholstery-sofa-free-combination-cloud-sectional-couch-with-l-shape-chaise-deep-seat-modular-sofa-for-living-room-bedroom-w111681030.html?piid=1216171221')

    """
    You don't have permission to access "http://www.crateandbarrel.com/lounge-93-sofa/s505694" on this server.
    Reference #18.934ddb17.1760392128.415e1240

    https://errors.edgesuite.net/18.934ddb17.1760392128.415e1240
    """
    time.sleep(2)  # Let the page load


    ## RETRIVING HTML
    html = driver.page_source

    # --- Optional: save HTML to a file ---
    with open("page.html", "w", encoding="utf-8") as f:
        f.write(html)

    matches = re.findall(r'<(?:img\b[^>]*?>|(?:td|span)\b[^>]*?>.*?</(?:td|span)>)', html, re.IGNORECASE | re.DOTALL)

    html = (' '.join(matches))
    
    with open('filtered_page.html', 'w', encoding='utf-8') as f2:
        f2.write(html)
    
    # Use RAG to analyze the product
    print("Analyzing product with RAG...")
    analysis_result = analyze_product_with_rag(html)
    print("\n" + "="*50)
    print("PRODUCT ANALYSIS RESULT:")
    print("="*50)
    print(analysis_result)
    print("="*50)

    time.sleep(20)  # Wait to see the results
    driver.quit()

if __name__ == "__main__":
    main()