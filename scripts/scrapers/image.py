import os
import sys

# Add project root to system path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added {PROJECT_ROOT} to Python path")

# Import database connection
from backend.api.database import get_supabase_client

# Constants
MAX_WORKERS = 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STICKMAN_URL = "https://static1.cbrimages.com/wordpress/wp-content/uploads/2021/01/Captain-Rocks.jpg"

def update_image_urls():
    """
    Updates all fighter image URLs to use the placeholder image.
    Processes fighters in batches to avoid API rate limits.
    """
    try:
        # Initialize database connection
        supabase = get_supabase_client()
        
        # Fetch all fighters using pagination to get beyond the 1000 record limit
        page_size = 1000
        all_fighters = []
        page = 0
        
        while True:
            # Fetch a page of fighters
            response = supabase.table('fighters') \
                .select('fighter_name') \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()
            
            # Add fighters to our list
            fighters_page = response.data
            all_fighters.extend(fighters_page)
            
            # If we got fewer results than the page size, we've reached the end
            if len(fighters_page) < page_size:
                break
                
            # Move to next page
            page += 1
        
        print(f"[INFO] Found {len(all_fighters)} fighters in the database to process.")
        
        # Process in batches
        batch_size = 50
        success_count = 0
        
        for i in range(0, len(all_fighters), batch_size):
            batch = all_fighters[i:i+batch_size]
            for fighter in batch:
                fighter_name = fighter['fighter_name']
                try:
                    # Update fighter record
                    response = supabase.table('fighters') \
                        .update({'image_url': STICKMAN_URL}) \
                        .eq('fighter_name', fighter_name) \
                        .execute()
                    
                    if response.data:
                        success_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to update {fighter_name}: {str(e)}")
            
            print(f"[INFO] Processed {min(i + batch_size, len(all_fighters))}/{len(all_fighters)} fighters...")
        
        print(f"[DONE] Successfully updated image URLs for {success_count} fighters with stickman placeholder")
    except Exception as e:
        print(f"[ERROR] Failed to update image URLs: {str(e)}")

if __name__ == "__main__":
    update_image_urls()