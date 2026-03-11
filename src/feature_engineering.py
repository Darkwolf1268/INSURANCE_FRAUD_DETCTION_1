def create_features(df):

    df["claim_to_premium_ratio"] = df["claim_amount"] / (df["policy_annual_premium"] + 1)

    return df
