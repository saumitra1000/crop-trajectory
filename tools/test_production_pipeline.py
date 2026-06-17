import pytest
import numpy as np
import json
import os
import joblib
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.inference_driver import normalize_to_coordinate_ring, generate_interior_points, predict_live_lpis_parcel
from tools.batch_geojson_classifier import execute_batch_geojson_classification, calculate_polygon_perimeter

# ===========================================================================
# 1. GEOMETRY NORMALIZER & CENTROID TEST CASES
# ===========================================================================
def test_normalize_standard_polygon():
    poly = [[-7.915, 53.342], [-7.910, 53.342], [-7.910, 53.347], [-7.915, 53.347], [-7.915, 53.342]]
    clean = normalize_to_coordinate_ring(poly)
    assert len(clean) == 5
    assert float(clean[0][0]) == -7.915

def test_normalize_nested_geojson_polygon():
    nested_poly = [[[-7.915, 53.342], [-7.910, 53.342], [-7.910, 53.347], [-7.915, 53.347], [-7.915, 53.342]]]
    clean = normalize_to_coordinate_ring(nested_poly)
    assert len(clean) == 5
    assert float(clean[0][0]) == -7.915

def test_normalize_multipolygon():
    multi_poly = [[[[ -7.915, 53.342 ], [ -7.910, 53.342 ], [ -7.910, 53.347 ], [ -7.915, 53.347 ], [ -7.915, 53.342 ]]]]
    clean = normalize_to_coordinate_ring(multi_poly)
    assert len(clean) == 5

def test_normalize_empty_and_invalid_geometry():
    assert normalize_to_coordinate_ring([]) == []
    assert normalize_to_coordinate_ring("NOT_A_LIST") == []

def test_generate_interior_points_safety():
    pts = generate_interior_points([])
    assert len(pts) == 3
    assert pts[0][0] == 52.14  # Dynamic fallback latitude anchor

# ===========================================================================
# 2. INFERENCE DRIVER & MULTI-TIER ROUTING TEST CASES
# ===========================================================================
@patch('joblib.load')
@patch('tools.inference_driver.extract_fusion_features')
def test_tier1_automated_delivery(mock_extract, mock_joblib):
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[4]])  # 2D CatBoost index array format
    mock_model.predict_proba.return_value = np.array([[0.05, 0.05, 0.05, 0.05, 0.80, 0.05, 0.05]]) # 80% certainty
    
    mock_le = Mock()
    mock_le.inverse_transform.return_value = ["Grassland"]
    
    mock_joblib.side_effect = [mock_model, mock_le, list(range(30))]
    mock_extract.return_value = dict(
        monthly_ndvi={str(m): 0.5 for m in range(1, 13)},
        monthly_ndre={str(m): 0.4 for m in range(1, 13)},
        monthly_vh={str(m): -15.0 for m in range(1, 13)},
        monthly_vv={str(m): -10.0 for m in range(1, 13)},
        n_sar=22, n_ndvi=8
    )
    
    poly = [[-7.915, 53.342], [-7.910, 53.342], [-7.910, 53.347]]
    pred, conf = predict_live_lpis_parcel(poly, area_ha=3.8, perimeter_m=320)
    
    assert conf == 0.80
    assert "Grassland" in str(pred)

@patch('joblib.load')
@patch('tools.inference_driver.extract_fusion_features')
def test_tier3_gated_rejection(mock_extract, mock_joblib):
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[2]])
    mock_model.predict_proba.return_value = np.array([[0.10, 0.10, 0.25, 0.15, 0.10, 0.10, 0.20]]) # 25% certainty
    
    mock_le = Mock()
    mock_le.inverse_transform.return_value = ["Oats"]
    
    mock_joblib.side_effect = [mock_model, mock_le, list(range(30))]
    mock_extract.return_value = dict(
        monthly_ndvi={str(m): 0.5 for m in range(1, 13)},
        monthly_ndre={str(m): 0.4 for m in range(1, 13)},
        monthly_vh={str(m): -15.0 for m in range(1, 13)},
        monthly_vv={str(m): -10.0 for m in range(1, 13)},
        n_sar=22, n_ndvi=8
    )
    
    poly = [[-7.915, 53.342], [-7.910, 53.342], [-7.910, 53.347]]
    pred, conf = predict_live_lpis_parcel(poly, area_ha=3.8, perimeter_m=320)
    
    assert "Unknown" in str(pred)
    assert conf == 0.25

# ===========================================================================
# 3. BATCH PROCESSOR SCHEMA VALIDATION TESTS
# ===========================================================================
@patch('tools.batch_geojson_classifier.predict_live_lpis_parcel')
def test_batch_geojson_schema_validation(mock_predict):
    mock_predict.return_value = ("Grassland", 0.95)
    
    mock_input = "/tmp/test_input.geojson"
    mock_output = "/tmp/test_output.geojson"
    
    mock_input_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"parcel_id": "IE-001", "area_ha": 2.5},
            "geometry": {"type": "Polygon", "coordinates": [[[-7.9, 53.3], [-7.8, 53.3], [-7.8, 53.4], [-7.9, 53.3]]]}
        }]
    }
    
    with open(mock_input, "w") as f:
        json.dump(mock_input_data, f)
        
    execute_batch_geojson_classification(mock_input, mock_output)
    
    assert os.path.exists(mock_output)
    with open(mock_output, "r") as f:
        out_data = json.load(f)
        
    out_feat = out_data["features"][0]
    assert "crop_prediction" in out_feat["properties"]
    assert out_feat["properties"]["delivery_tier"] == "Tier 1: Automated Delivery"
    assert "inference_confidence" in out_feat["properties"]
    
    # Cleanup pipeline temporary data
    for p in [mock_input, mock_output]:
        if os.path.exists(p):
            os.remove(p)
