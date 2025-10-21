# PreviewAR

## Overall pipeline overview:
###  (Decision agent) There is going to be a decision agent that will be determining what essential information there is in a given URL:
- It can choose either to have an API decision or to have a web scraping decision 
- If it is a API decision:	
    - Chosen decision (API agent)
    - Company name 
    - Identifier (e.g. asin, item_id)
    - Api call link 
- If it is a web scrapping decision:
    - Chosen decision (web scrapping agent)
    - Company name
    - Html code
### (API agents)
#### Amazon api
- Currently, there are issues with the use of the standard APIs so I’ve been using a tool 
  called rainforest API, which has allowed me to
  bypass the authentication issues/endpoint issues that typically arise 
- I’m going to have a call with the person in charge to also increase the number of tokens 
  available before I have to pay for the api 
- Credentials necessary: 
    - Amazon PAAPI key for auth with Amazon 
    - Rainforest API key
- Returns the json from Amazon’s API
#### Ebay api 
- Two-step process for getting information from the eBay API. 
    1. Need to get the auth token from ebay with the client_id and the client_secret. 
    2. Once the token is retrieved, you can then use the token to retrieve the rest of the information
- Caveat to the retrieval of information, some objects in eBay are group objects. 
    - Group case: 
        - You will need to use a group id 
        - Different endpoint needs to be used 
    - Individual case:
        - You will need to use a legacy_id
        - Use the legacy_id endpoint 
        - This is easier than having to use a restful api endpoint since the restful api version of 
          the legacy_id is not readily available
- Return the json in either a group or individual format. 
    - If group, let the user choose which option in the group
    - If individual, next step 


### (Web scraping agent)
- This is something that needs to be developed
- Error messages will be displayed when there is an error in the scraping 
- If no error, the data will be scrapped
### (Top image agent)
- Choose the top image from a set of images in the json
- Given a list of image urls, have GPT-5 analyze the images with a given set of criteria to determine 
  the best image to choose for DIS 
### (DIS (dichotomous image segmentation))
- For the following repo: https://github.com/park2003/DIS 
- Takes the image mask, and with the image mask, it will then use the bits for the segmentation mask 
  to mask out the parts of the image not relevant to the main part of the image. 
### (Hunyuan 3D) 
- Run into issues on AnyDesk where whenever i do something that involves the running of stuff that is 
  credentials-oriented, AnyDesk crashes
- I am planning to complete credentials-oriented setup at my aunts and uncles' place on the weekend, so 
  that everything is set up for the rest of next week regarding Hunyuan 3D setup 
- Set up required involves: 
    - CUDA toolkit set up for 12.6
    - Setting up cuda toolkit variables 
    - Check with: nvcc –version
    - Install pytorch for cuda 
    - Make sure that MSVC is installed
### (Unity + C#) HoloLens 2 Setup
- Deploy the model into a scene in Unity for the user
