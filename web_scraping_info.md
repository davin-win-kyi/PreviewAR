# Website Scraping Configuration

## Amazon
### Initial Cleanup
- **Cookie banner:** `#sp-cc-accept`, `input#sp-cc-accept`
- **Sign-in modal:** `#nav-signin-tooltip .nav-action-button`
- **Generic close buttons:** `button[aria-label*='Close']`, `button[name='cancel']`

### Scrape Info
- **Product dimensions:** `#productDetails_techSpec_section_1`, `#productDetails_detailBullets_sections1`
- **Images:** `#landingImage`, `.imgTagWrapper img`

---

## IKEA
### Initial Cleanup
- **Cookie banner:** `button[data-testid='accept-cookies-button']`, `button[data-test='cookie-accept-all-button']`
- **Region modal:** `button[aria-label*='close']`
- **Generic close buttons:** `div[class*='modal'] button`, `div[class*='cookie'] button`

### Scrape Info
- **Product dimensions:** `.product-pip__dimensions`, `.product-dimensions__table`
- **Images:** `.range-revamp-media-grid__image img`, `img[data-test='product-pip-gallery-image']`

---

## eBay
### Initial Cleanup
- **Cookie banner:** `button[aria-label='Got it']`, `button[aria-label='No thanks']`
- **Newsletter modal:** `button[aria-label*='close']`, `button[data-test-id='dialog-close']`

### Scrape Info
- **Product dimensions:** `.itemAttr td:contains('Dimensions') + td`, `.itemAttr td:contains('Size') + td`
- **Images:** `#icImg`, `.img.img.img500`

---

## Wayfair
### Initial Cleanup
- **Newsletter modal:** `button[data-enzyme-id='email-subscription-close']`, `button[data-testid='modal-close']`
- **Generic close buttons:** `button[aria-label='Close']`, `div[role='dialog'] button`

### Scrape Info
- **Product dimensions:** `.product-dimensions`, `.dimensions-section`
- **Images:** `.media-gallery img`, `img[data-testid='product-image']`
