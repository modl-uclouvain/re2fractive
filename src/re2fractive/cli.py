def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "campaign", help="The campaign ID to start or load.", type=str, default=None
    )
    parser.add_argument(
        "--march",
        help="Whether to continue running the campaign.",
        type=bool,
        default=False,
    )

    campaign_id = parser.parse_args().campaign
    march = parser.parse_args().march
    if os.environ.get("RE2FRACTIVE_CAMPAIGN_ID") and campaign_id is not None:
        raise RuntimeError("Campaign ID already set in environment variable.")

    os.environ["RE2FRACTIVE_CAMPAIGN_ID"] = campaign_id

    from re2fractive.campaign import Campaign

    campaign = Campaign.load()

    if march:
        campaign.march()

    else:
        breakpoint()
