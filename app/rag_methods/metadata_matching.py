from fuzzywuzzy import process

def fuzzy_match_all(value, candidates, threshold=80):
    """
    Returns a list of all candidate matches with a score above the threshold.
    If no candidate meets the threshold, returns the original value.
    """
    if value == "-":
        return value
    matches = process.extractBests(value, candidates, score_cutoff=threshold)
    if matches:
        return [match[0] for match in matches]
    else:
        return value


def fuzzy_match_one(value, candidates, threshold=80):
    """
    Returns the best candidate match with a score above the threshold.
    If no candidate meets the threshold, returns the original value.
    """
    if value == "-":
        return value
    match = process.extractOne(value, candidates, score_cutoff=threshold)
    return match[0] if match else value


def match_metadata_all(extracted_metadata, allowed_values):
    """
    Matches string fields using fuzzy matching that returns all candidates above the threshold.

    Parameters:
      extracted_metadata (dict): The metadata extracted from the query.
      allowed_values (dict): Allowed values for each field, e.g.:
          {
            "variety": [...],
            "designation": [...],
            "country": [...],
            "province": [...],
            "region_1": [...],  # Optional
            "wine_color": [...],
          }

    Returns:
      dict: A new dictionary with normalized metadata.
            For string fields, the value will be a list of matching candidates if more than one match is found.
            Numeric fields are passed unchanged.
    """
    matched = {}

    matched["min_price"] = extracted_metadata.get("min_price", -1)
    matched["max_price"] = extracted_metadata.get("max_price", -1)
    matched["points"] = extracted_metadata.get("points", -1)
    matched["min_vintage"] = extracted_metadata.get("min_vintage", -1)
    matched["max_vintage"] = extracted_metadata.get("max_vintage", -1)

    combined_candidates = list(set(allowed_values.get("variety", []) + allowed_values.get("designation", [])))
    matched["variety_designation"] = fuzzy_match_all(extracted_metadata.get("variety_designation", "-"),
                                                     combined_candidates)

    allowed_provinces = allowed_values.get("province", [])
    if "region_1" in allowed_values:
        allowed_provinces = list(set(allowed_provinces) | set(allowed_values["region_1"]))
    matched["province"] = fuzzy_match_all(extracted_metadata.get("province", "-"), allowed_provinces)

    matched["country"] = fuzzy_match_one(extracted_metadata.get("country", "-"), allowed_values.get("country", []))
    matched["wine_color"] = fuzzy_match_one(extracted_metadata.get("wine_color", "-"),
                                            allowed_values.get("wine_color", []))

    return matched


# allowed_values = {
#     "variety": df["variety"].dropna().unique().tolist(),
#     "designation": df["designation"].dropna().unique().tolist(),
#     "country": df["country"].dropna().unique().tolist(),
#     "province": df["province"].dropna().unique().tolist(),
#     "region_1": df["region_1"].dropna().unique().tolist(),
#     "wine_color": df["wine_color"].dropna().unique().tolist()
# }


def metadata_matches(doc_meta, constraints):
    """
    Checks if the document's metadata matches the given constraints.

    For numeric fields (e.g., price, points, vintage):
      - If the constraint is '-', ignore it.
      - For "max_price", require doc_meta["price"] <= constraint.
      - For "min_price", require doc_meta["price"] >= constraint.
      - For "points", require doc_meta["points"] >= constraint.
      - For "max_vintage", require doc_meta["vintage"] <= constraint.
      - For "min_vintage", require doc_meta["vintage"] >= constraint.

    For string fields:
      - For "variety_designation":
          If the constraint is '-', ignore it.
          Otherwise, require that either doc_meta["variety"] or doc_meta["designation"]
          (case-insensitive) exactly matches the constraint.
      - For "country" and "wine_color":
          If the constraint is '-', ignore it.
          Otherwise, require an exact match.
      - For "province":
          If the constraint is '-', ignore it.
          Otherwise, require doc_meta["province"] or doc_meta["region_1"] to match
    """
    if constraints.get("max_price", '-') != '-':
        if doc_meta.get("price", float('inf')) > constraints["max_price"]:
            return False
    if constraints.get("min_price", '-') != '-':
        if doc_meta.get("price", -float('inf')) < constraints["min_price"]:
            return False
    if constraints.get("points", '-') != '-':
        if doc_meta.get("points", -1) < constraints["points"]:
            return False
    if constraints.get("max_vintage", '-') != '-':
        if doc_meta.get("vintage", float('inf')) > constraints["max_vintage"]:
            return False
    if constraints.get("min_vintage", '-') != '-':
        if doc_meta.get("vintage", -float('inf')) < constraints["min_vintage"]:
            return False

    cons_val = constraints.get("variety_designation", "-")
    if cons_val != "-":
        cons_val = str(cons_val).strip().lower()
        doc_variety = doc_meta.get("variety", "").strip().lower()
        doc_designation = doc_meta.get("designation", "").strip().lower()
        if (doc_variety != cons_val) and (doc_designation != cons_val):
            return False

    for field in ["country", "wine_color"]:
        cons_val = constraints.get(field, "-")
        if cons_val != "-":
            cons_val = str(cons_val).strip().lower()
            doc_val = doc_meta.get(field, "").strip().lower()
            if doc_val != cons_val:
                return False

    cons_val = constraints.get("province", "-")
    if cons_val != "-":
        cons_val = str(cons_val).strip().lower()
        doc_province = doc_meta.get("province", "").strip().lower()
        doc_region = doc_meta.get("region_1", "").strip().lower()
        if (doc_province != cons_val) and (doc_region != cons_val):
            return False

    return True