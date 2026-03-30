"""
Flask API for Hierarchical Plant Disease Detection
===================================================

A production-ready REST API for the hierarchical plant disease detection system.

Endpoints:
    POST /predict - Single image prediction
    POST /predict-batch - Batch prediction
    GET /health - Health check
    GET /info - System information

Usage:
    python api.py

Then send requests:
    curl -X POST -F "image=@test.jpg" http://localhost:5000/predict
"""