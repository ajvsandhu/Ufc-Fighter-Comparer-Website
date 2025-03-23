import sqlite3
import os

# Constants
MAX_WORKERS = 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../data/ufc_fighters.db"))
STICKMAN_URL = "https://static1.cbrimages.com/wordpress/wp-content/uploads/2021/01/Captain-Rocks.jpg"

def update_image_urls():
    """
    Add image_url column if it doesn't exist and update image URLs for all fighters 
    to use the stickman placeholder URL, preserving existing data.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Check if image_url column exists, if not, add it
    cur.execute("PRAGMA table_info(fighters)")
    columns = [col[1] for col in cur.fetchall()]
    if 'image_url' not in columns:
        print("[INFO] Adding image_url column to fighters table")
        cur.execute("ALTER TABLE fighters ADD COLUMN image_url TEXT")
    
    # Get all fighter names from the database
    cur.execute("SELECT fighter_name FROM fighters")
    fighters = cur.fetchall()
    
    print(f"[INFO] Found {len(fighters)} fighters in the database to process.")
    
    # Update each fighter's image_url to the stickman URL
    for fighter in fighters:
        fighter_name = fighter[0]
        cur.execute("""
            UPDATE fighters 
            SET image_url = ? 
            WHERE fighter_name = ?
        """, (STICKMAN_URL, fighter_name))
    
    conn.commit()
    conn.close()
    print(f"[DONE] Updated image URLs for {len(fighters)} fighters in '{DB_PATH}' with stickman placeholder")

if __name__ == "__main__":
    update_image_urls()