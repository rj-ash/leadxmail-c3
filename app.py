from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
from typing import List, Optional
from lead_match import evaluate_product_relevancy, evaluate_multiple_leads
import uvicorn

app = FastAPI(
    title="Lead Match API",
    description="API for evaluating product-lead matching scores",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://flow-forge-campaigns.lovable.app"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class LeadInfo(BaseModel):
    name: str
    lead_id: int
    experience: str
    education: str
    company: str
    company_overview: str
    company_industry: str

class ProductDetails(BaseModel):
    details: str

class MultipleLeadsRequest(BaseModel):
    product_details: str
    leads: List[LeadInfo]

class SingleLeadResponse(BaseModel):
    lead_id: int
    relevance_score: float
    lead_name: str

class LeadScore(BaseModel):
    lead_id: int
    relevance_score: float

class MultipleLeadsResponse(BaseModel):
    results: List[LeadScore]

@app.post("/evaluate-single", response_model=SingleLeadResponse)
async def evaluate_single_lead(product: ProductDetails, lead: LeadInfo):
    """
    Evaluate the relevancy of a product for a single lead.
    Returns a score between 0 and 10.
    """
    try:
        score = evaluate_product_relevancy(product.details, lead.dict())
        return SingleLeadResponse(
            lead_id=lead.lead_id,
            relevance_score=score,
            lead_name=lead.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-multiple", response_model=MultipleLeadsResponse)
async def evaluate_multiple_leads_endpoint(request: MultipleLeadsRequest):
    """
    Evaluate the relevancy of a product for multiple leads.
    Returns scores for each lead between 0 and 10.
    """
    try:
        results = evaluate_multiple_leads(request.product_details, [lead.dict() for lead in request.leads])
        return MultipleLeadsResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 