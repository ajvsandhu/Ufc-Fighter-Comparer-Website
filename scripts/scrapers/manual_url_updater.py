#!/usr/bin/env python
"""
Manual URL Updater for the fighters table.
Allows manual input of image_url and tap_link for a specified fighter.
"""

import os
import sys
import logging
import sqlite3

# Fix import by correctly adding the project root to sys.path
# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added {PROJECT_ROOT} to Python path")

# Now import from backend should work
from backend.api.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fighter_exists(fighter_name: str) -> bool:
    """Checks if a fighter exists in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fighters WHERE fighter_name = ?", (fighter_name,))
    exists = cur.fetchone()[0] > 0
    conn.close()
    return exists

def update_fighter_urls(fighter_name: str, image_url: str, tap_link: str):
    """Updates the fighter's image_url and tap_link in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE fighters SET image_url = ?, tap_link = ? WHERE fighter_name = ?",
            (image_url, tap_link, fighter_name)
        )
        if cur.rowcount > 0:
            conn.commit()
            logger.info(f"Successfully updated {fighter_name}: image_url={image_url}, tap_link={tap_link}")
        else:
            logger.warning(f"No rows updated for {fighter_name}. Check if the name matches exactly.")
    except sqlite3.Error as e:
        logger.error(f"Database error while updating {fighter_name}: {e}")
    finally:
        conn.close()

def main():
    """Main function to handle manual URL input."""
    print("Manual URL Updater for Fighters Table")
    print("Enter details below. Press Ctrl+C to exit at any time.\n")

    while True:
        try:
            # Get fighter name
            fighter_name = input("Enter fighter name (exact match required): ").strip()
            if not fighter_name:
                logger.warning("Fighter name cannot be empty. Try again.")
                continue

            # Check if fighter exists
            if not fighter_exists(fighter_name):
                logger.warning(f"'{fighter_name}' not found in the database. Please check the name and try again.")
                continue

            # Get image URL
            image_url = input("Enter image_url (or press Enter to skip): ").strip()
            if not image_url:
                image_url = None  # Allow skipping with empty input

            # Get Tapology link
            tap_link = input("Enter tap_link (or press Enter to skip): ").strip()
            if not tap_link:
                tap_link = None  # Allow skipping with empty input

            # Confirm before updating
            print(f"\nConfirm update for {fighter_name}:")
            print(f"image_url: {image_url if image_url else 'No change'}")
            print(f"tap_link: {tap_link if tap_link else 'No change'}")
            confirm = input("Proceed? (y/n): ").lower().strip()
            if confirm != 'y':
                logger.info("Update cancelled.")
                continue

            # Perform the update
            update_fighter_urls(fighter_name, image_url, tap_link)

            # Ask if user wants to continue
            more = input("\nUpdate another fighter? (y/n): ").lower().strip()
            if more != 'y':
                logger.info("Exiting manual updater.")
                break

        except KeyboardInterrupt:
            logger.info("\nUser interrupted. Exiting.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            continue

if __name__ == "__main__":
    main()