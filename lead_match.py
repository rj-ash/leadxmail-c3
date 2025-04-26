from dotenv import load_dotenv
import os
from typing import List
from langchain_openai import ChatOpenAI
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
import json


def evaluate_product_relevancy(product_details: str, lead_info: dict) -> float:
    """
    Evaluates the relevancy of a product for a specific lead based on their LinkedIn profile.
    
    Args:
        product_details (str): Detailed description of the product
        lead_info (dict): Dictionary containing lead information in the format:
            {
                'name': str,
                'lead_id': int,
                'experience': str,
                'education': str,
                'company': str,
                'company_overview': str,
                'company_industry': str
            }
    
    Returns:
        float: A score between 0 and 10 indicating the relevancy of the product for this lead
    """
    # Load environment variables
    load_dotenv()

    # Ensure the API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

    # Initialize the OpenAI model with optimized settings
    model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.3,  # Lower temperature for more consistent results
        max_tokens=10,  # Limit response size
    )

    # Construct the prompt
    prompt = f"""
    Analyze the relevancy of the following product for this LinkedIn lead. Consider the company they work for (Use your knowledge base to analyse the company and whether it is a good fit for the product), their experience and background.
    
    Product Details:
    ```{product_details}```
    
    Lead Information:
    ```{lead_info}```
    
    Evaluate the match between the product and the lead's profile. Consider:
    1. How well the product aligns with their current role and responsibilities
    2. Whether their skills and experience make them a good fit for this product
    3. If their industry/domain matches the product's target market
    4. Their level of seniority and decision-making authority (If they are a new employee or intern, they may not be the decision-maker. Keep the score low for them)
    
    Return ONLY a single number between 0 and 10 (with one decimal place) representing the relevancy score, where:
    - 0-3: Poor match
    - 4-6: Moderate match
    - 7-8: Good match
    - 9-10: Excellent match
    
    Example response: 7.5
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        after=lambda retry_state: time.sleep(2)  # Additional delay between retries
    )
    def _invoke_model():
        try:
            result = model.invoke(prompt)
            return result
        except (RateLimitError, APIError) as e:
            print(f"Rate limit hit, retrying after delay... Error: {str(e)}")
            raise
        except Exception as e:
            print(f"Error invoking model: {str(e)}")
            raise

    try:
        # Invoke the model with retry logic
        result = _invoke_model()
        
        # Extract the score from the response
        try:
            score = float(result.content.strip())
            # Ensure the score is between 0 and 10
            return max(0, min(10, score))
        except ValueError:
            return 0.0  # Return 0 if we can't parse the score
    except Exception as e:
        print(f"Failed to evaluate product relevancy: {str(e)}")
        return 0.0  # Return 0 on any error


def evaluate_multiple_leads(product_details: str, leads_data: List[dict]) -> List[dict]:
    """
    Evaluates product relevancy for multiple leads and returns their lead_ids and scores.
    
    Args:
        product_details (str): Detailed description of the product
        leads_data (List[dict]): List of dictionaries containing lead information, where each dict has:
            {
                'name': str,
                'lead_id': int,
                'experience': str,
                'education': str,
                'company': str,
                'company_overview': str,
                'company_industry': str
            }
    
    Returns:
        List[dict]: List of dictionaries containing:
            {
                'lead_id': int,
                'relevance_score': float
            }
    """
    # Initialize list to store results
    results = []
    
    # Process each lead with delay between evaluations
    for i, lead in enumerate(leads_data):
        try:
            # Evaluate product relevancy for this lead
            score = evaluate_product_relevancy(product_details, lead)
            
            # Add result to list
            results.append({
                'lead_id': lead['lead_id'],
                'relevance_score': score
            })
            
            # Add delay between evaluations to avoid rate limits
            if i < len(leads_data) - 1:  # Don't delay after the last lead
                time.sleep(2)  # 2-second delay between evaluations
                
        except Exception as e:
            print(f"Error processing lead {lead.get('name', 'Unknown')}: {str(e)}")
            results.append({
                'lead_id': lead.get('lead_id', 0),
                'relevance_score': 0.0
            })
    
    # Return the results
    return results



# product_details = """
# AI-Powered Sales Automation Platform
#         Features:
#         - Automated lead scoring and prioritization
#         - AI-driven email personalization
#         - Sales pipeline analytics
#         - CRM integration
#         - Team collaboration tools
        
#         Target Market:
#         - B2B SaaS companies
#         - Sales teams of 10+ members
#         - Mid-market to enterprise companies
#         - Companies with complex sales processes

# """

# lead_info_1 = {
#             'name': 'John Doe',
#             'experience': 'Teacher at a school',
#             'education': 'MBA in Business Administration',
#             'company': 'Central academy School',
#             'company_overview': 'School for children',
#             'company_industry': 'Education'
#         }


# score = evaluate_product_relevancy(product_details, lead_info_1)
# print(score)
# print(type(score))
        