def evaluate_crop_consensus(dafm_declaration, sar_heuristic, catboost_prediction):
    """
    🎯 Production Consensus Engine:
    Cross-references three separate signal matrices to protect Cube Earth
    against false positives and spatial classification drifts.
    """
    # Standardize input formats to eliminate case variations
    d_clean = str(dafm_declaration).strip().lower()
    s_clean = str(sar_heuristic).strip().lower()
    c_clean = str(catboost_prediction).strip().lower()
    
    # Tier 1: Absolute Alignment Across All Three Inputs
    if d_clean == s_clean == c_clean:
        return "Strong Consensus"
        
    # Tier 2: Model aligns with at least one administrative/heuristic anchor
    if c_clean == d_clean or c_clean == s_clean:
        return "Moderate Consensus"
        
    # Tier 3: Divergence Between anchors (Requires human audit)
    if d_clean == s_clean and c_clean != d_clean:
        return "Divergent Model (Administrative Alignment)"
        
    return "No Consensus (Review Required)"
