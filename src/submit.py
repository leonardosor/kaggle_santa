"""
Submit solution to Kaggle Santa 2025 competition
"""

import argparse
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def submit_solution(
    csv_path: str,
    competition: str = "santa-2025",
    message: str = "Optimized packing solution"
):
    """
    Submit the solution to Kaggle

    Args:
        csv_path: Path to submission CSV file
        competition: Kaggle competition name
        message: Submission description

    Returns:
        bool: True if successful, False otherwise
    """

    print("=" * 70)
    print("KAGGLE SUBMISSION")
    print("=" * 70)

    # Check file exists
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] File not found: {csv_path}")
        return False

    print(f"\n1. Submission file: {csv_path}")
    print(f"   File size: {os.path.getsize(csv_path) / 1024:.2f} KB")

    # Initialize API
    print("\n2. Authenticating with Kaggle...")
    try:
        api = KaggleApi()
        api.authenticate()
        print("   [OK] Authenticated")
    except Exception as e:
        print(f"   [ERROR] Authentication failed: {e}")
        print("\n   Setup instructions:")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. Click 'Create New API Token'")
        print("   3. Place kaggle.json in ~/.kaggle/")
        return False

    # Submit
    print(f"\n3. Submitting to competition: {competition}")
    print(f"   Message: {message}")

    try:
        result = api.competition_submit(
            file_name=csv_path,
            message=message,
            competition=competition
        )
        print("   [SUCCESS] Submission uploaded!")
        print(f"\n   Result: {result}")

        print("\n" + "=" * 70)
        print("SUBMISSION SUCCESSFUL!")
        print("=" * 70)
        print("\nCheck your submission at:")
        print(f"https://www.kaggle.com/competitions/{competition}/submissions")

        return True

    except Exception as e:
        print(f"   [ERROR] Submission failed: {e}")

        print("\n   Possible issues:")
        print("   1. Competition rules not accepted")
        print(f"      - Go to: https://www.kaggle.com/competitions/{competition}")
        print("      - Click 'Join Competition' and accept rules")
        print("\n   2. API token invalid")
        print("      - Generate new token at: https://www.kaggle.com/settings")
        print("\n   3. Submission format incorrect")
        print("      - Check sample_submission.csv format")

        print("\n" + "=" * 70)
        print("MANUAL SUBMISSION INSTRUCTIONS")
        print("=" * 70)
        print(f"\n1. Go to: https://www.kaggle.com/competitions/{competition}/submit")
        print(f"2. Upload file: {os.path.abspath(csv_path)}")
        print(f"3. Description: {message}")
        print("4. Click 'Submit'")

        return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit solution to Kaggle Santa 2025 competition"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to submission CSV file"
    )
    parser.add_argument(
        "-c", "--competition",
        type=str,
        default="santa-2025",
        help="Competition name (default: santa-2025)"
    )
    parser.add_argument(
        "-m", "--message",
        type=str,
        default="Optimized packing solution",
        help="Submission message"
    )

    args = parser.parse_args()

    success = submit_solution(args.csv_path, args.competition, args.message)

    if not success:
        print("\n\nIf API submission isn't working, you can submit manually!")
        exit(1)


if __name__ == "__main__":
    main()
