from fuzzywuzzy import process, fuzz
from vectorstore.create_vectorstore import create_documents


def fuzzy_match_all(value, candidates, threshold=80):
    if value == "-":
        return value
    matches = process.extractBests(value, candidates, score_cutoff=threshold)
    if matches:
        return [match[0] for match in matches]
    else:
        return value


def fuzzy_match_one(value, candidates, threshold=80):
    if value == "-":
        return value
    match = process.extractOne(value, candidates, score_cutoff=threshold)
    return match[0] if match else value


def match_metadata_all(extracted_metadata, allowed_values):
    matched_positive = {}
    for key in ["min_price", "max_price", "points", "min_vintage", "max_vintage"]:
        matched_positive[key] = extracted_metadata["positive"].get(key, -1)

    combined_candidates = list(set(allowed_values.get("variety", []) + allowed_values.get("designation", [])))
    allowed_provinces = allowed_values.get("province", [])
    if "region_1" in allowed_values:
        allowed_provinces += allowed_values["region_1"]
    allowed_provinces = list(set(allowed_provinces))

    matched_positive["variety_designation"] = fuzzy_match_all(
        extracted_metadata["positive"].get("variety_designation", "-"), combined_candidates
    )
    matched_positive["province"] = fuzzy_match_all(
        extracted_metadata["positive"].get("province", "-"), allowed_provinces
    )
    matched_positive["country"] = fuzzy_match_one(
        extracted_metadata["positive"].get("country", "-"), allowed_values.get("country", [])
    )
    matched_positive["wine_color"] = fuzzy_match_one(
        extracted_metadata["positive"].get("wine_color", "-"), allowed_values.get("wine_color", [])
    )

    matched_negative = {}
    matched_negative["variety_designation"] = fuzzy_match_one(
        extracted_metadata["negative"].get("variety_designation", "-"), combined_candidates
    )
    matched_negative["province"] = fuzzy_match_one(
        extracted_metadata["negative"].get("province", "-"), allowed_provinces
    )
    matched_negative["country"] = fuzzy_match_one(
        extracted_metadata["negative"].get("country", "-"), allowed_values.get("country", [])
    )
    matched_negative["wine_color"] = fuzzy_match_one(
        extracted_metadata["negative"].get("wine_color", "-"), allowed_values.get("wine_color", [])
    )

    return {
        "positive": matched_positive,
        "negative": matched_negative
    }

def get_allowed_values(df):
    allowed_values = {
        "variety": df["variety"].dropna().unique().tolist(),
        "designation": df["designation"].dropna().unique().tolist(),
        "country": df["country"].dropna().unique().tolist(),
        "province": df["province"].dropna().unique().tolist(),
        "region_1": df["region_1"].dropna().unique().tolist(),
        "wine_color": df["wine_color"].dropna().unique().tolist()
    }
    return allowed_values


def metadata_matches(doc_meta, constraints):
    positive = constraints.get("positive", {})
    negative = constraints.get("negative", {})
    # --- Positive Constraints ---
    for key, limit in positive.items():
        if limit == "-":
            continue
        if key in ["min_price", "max_price", "min_vintage", "max_vintage", "points"]:
            value = doc_meta.get(key.replace("min_", "").replace("max_", ""), None)
            if value is None:
                return False
            if key.startswith("min_") and value < limit:
                return False
            if key.startswith("max_") and value > limit:
                return False
            if key == "points" and value < limit:
                return False
        elif key == "variety_designation":
            val1 = doc_meta.get("variety", "").lower()
            val2 = doc_meta.get("designation", "").lower()
            targets = [v.lower() for v in (limit if isinstance(limit, list) else [limit])]
            if not any(t in [val1, val2] for t in targets):
                return False

        elif key == "province":
            val1 = doc_meta.get("province", "").lower()
            val2 = doc_meta.get("region_1", "").lower()
            targets = [v.lower() for v in (limit if isinstance(limit, list) else [limit])]
            if not any(t in [val1, val2] for t in targets):
                return False

        else:
            val = str(doc_meta.get(key, "")).lower()
            target = str(limit).lower()
            if val != target:
                return False

    # --- Negative Constraints ---
    for key, bad_val in negative.items():
        if bad_val == "-":
            continue

        values = [v.lower() for v in (bad_val if isinstance(bad_val, list) else [bad_val])]

        if key == "variety_designation":
            val1 = doc_meta.get("variety", "").lower()
            val2 = doc_meta.get("designation", "").lower()
            if any(v in [val1, val2] for v in values):
                return False

        elif key == "province":
            val1 = doc_meta.get("province", "").lower()
            val2 = doc_meta.get("region_1", "").lower()
            if any(v in [val1, val2] for v in values):
                return False

        else:
            val = str(doc_meta.get(key, "")).lower()
            if val in values:
                return False

    return True

def get_similar_wine(df, wine_name):
    def match_wine_name_best(df, wine_name, threshold=80):
        titles = df['title'].dropna().unique().tolist()
        best_match = process.extractOne(wine_name, titles, scorer=fuzz.token_set_ratio)

        if best_match and best_match[1] >= threshold:
            return best_match[0]
        return None

    def get_document_by_title(df, target_title):
        target = target_title.strip().lower()

        df["title_normalized"] = df["title"].astype(str).str.strip().str.lower()

        filtered_df = df[df["title_normalized"] == target]

        if filtered_df.empty:
            print(f"[ERROR] No exact match for title: {target_title}")
            print("[INFO] Top fuzzy matches:")
            titles = df["title"].dropna().unique().tolist()
            print(process.extract(target_title, titles, limit=5))
            raise ValueError(f"Could not find document for matched title '{target_title}'")

        documents = create_documents(filtered_df)

        if not documents:
            raise ValueError(f"Document creation failed for matched title '{target_title}'")

        return documents[0]

    matched_wine_title = match_wine_name_best(df, wine_name)
    if not matched_wine_title:
        raise ValueError(f"No wine matched the title '{wine_name}' with sufficient confidence.")

    matched_wine_document = get_document_by_title(df, matched_wine_title)
    if not matched_wine_document:
        raise ValueError(f"Could not find document for matched title '{matched_wine_title}'")

    return matched_wine_document
