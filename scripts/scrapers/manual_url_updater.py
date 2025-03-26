#!/usr/bin/env python3
"""
Manual URL Updater

A utility for manually updating fighter image URLs and Tapology links in the database.
"""
import os
import sys

# Fix import paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.api.database import get_supabase_client
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("manual_url_updater.log")
    ]
)
logger = logging.getLogger("manual_url_updater")

def check_fighter_exists(fighter_name):
    """
    Verify fighter exists in database.
    """
    try:
        supabase = get_supabase_client()
        response = supabase.table('fighters') \
            .select('fighter_name') \
            .eq('fighter_name', fighter_name) \
            .execute()
        
        return response.data and len(response.data) > 0
    except Exception as e:
        logger.error(f"Fighter lookup failed: {str(e)}")
        return False

def update_fighter_image(fighter_name, image_url):
    """
    Update fighter's image URL.
    """
    try:
        supabase = get_supabase_client()
        response = supabase.table('fighters') \
            .update({'image_url': image_url}) \
            .eq('fighter_name', fighter_name) \
            .execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Updated image for {fighter_name}")
            return True
        else:
            logger.debug(f"Image update failed for {fighter_name}")
            return False
    except Exception as e:
        logger.error(f"Image update failed: {str(e)}")
        return False

def update_fighter_tap_link(fighter_name, tap_link):
    """
    Update fighter's Tapology link.
    """
    try:
        supabase = get_supabase_client()
        response = supabase.table('fighters') \
            .update({'tap_link': tap_link}) \
            .eq('fighter_name', fighter_name) \
            .execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Updated Tapology link for {fighter_name}")
            return True
        else:
            logger.debug(f"Link update failed for {fighter_name}")
            return False
    except Exception as e:
        logger.error(f"Link update failed: {str(e)}")
        return False

def main():
    """
    Interactive prompt for manually updating fighter URLs.
    """
    print("\nManual Fighter URL Updater")
    print("-------------------------")
    
    while True:
        fighter_name = input("\nEnter fighter name (or 'q' to quit): ").strip()
        if fighter_name.lower() == 'q':
            break
            
        if not check_fighter_exists(fighter_name):
            print(f"Fighter '{fighter_name}' not found in database")
            continue
            
        print("\nUpdate options:")
        print("1. Image URL")
        print("2. Tapology Link")
        print("3. Both")
        print("4. Skip")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1' or choice == '3':
            image_url = input("Enter new image URL: ").strip()
            if image_url:
                if update_fighter_image(fighter_name, image_url):
                    print("Image URL updated successfully")
                else:
                    print("Failed to update image URL")
                    
        if choice == '2' or choice == '3':
            tap_link = input("Enter new Tapology link: ").strip()
            if tap_link:
                if update_fighter_tap_link(fighter_name, tap_link):
                    print("Tapology link updated successfully")
                else:
                    print("Failed to update Tapology link")
                    
        if choice == '4':
            continue
            
        if choice not in ['1', '2', '3', '4']:
            print("Invalid choice")
            
    print("\nExiting URL updater")

if __name__ == "__main__":
    main()