"""
FASTAPI + GRADIO SERVING APPLICATION

The application provides a complete serving solution for Telco Customer Churn Prediction Model.

Architecture:
- FastAPI: High-performance REST API
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict