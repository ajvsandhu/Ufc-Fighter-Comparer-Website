from fastapi import APIRouter, Query
from Api.database import get_db_connection

router = APIRouter()

@router.get("/fighters")
def get_fighters(query: str = Query("", min_length=1)):
    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure the query only selects fighter names and allows partial matches
    cur.execute("SELECT fighter_name FROM fighters WHERE fighter_name LIKE ? ORDER BY fighter_name ASC", ("%" + query + "%",))
    fighters = [row[0] for row in cur.fetchall()]  # Extract fighter names

    conn.close()
    return {"fighters": fighters}

@router.get("/fighter/{name}")
def get_fighter_stats(name: str):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Fetch personal details
    cur.execute("SELECT * FROM fighters WHERE fighter_name = ?", (name,))
    fighter_info = cur.fetchone()

    # Fetch last 5 fights
    cur.execute("SELECT * FROM fighter_last_5_fights WHERE fighter_name = ?", (name,))
    last_5_fights = cur.fetchall()

    conn.close()

    if not fighter_info:
        return {"error": "Fighter not found"}

    return {
        "fighter_info": dict(fighter_info),
        "last_5_fights": [dict(fight) for fight in last_5_fights]
    }

