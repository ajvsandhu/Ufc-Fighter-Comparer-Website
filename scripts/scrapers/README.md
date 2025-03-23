# UFC Stats Scrapers

This directory contains various scrapers for UFC fighter data and statistics.

## UFC Fighter Stats Scraper (ssqlufc.py)

This script scrapes fighter information from the UFC Stats website and stores it in an SQLite database.

### Features

- Scrapes complete fighter profiles including records, physical stats, and performance metrics
- Properly formats data (percentages, decimal numbers, etc.)
- Alphabetical sorting by last name
- Smart database updates (only updates changed data)
- Preserves additional data columns
- Two operating modes: full update or recent fighters only

### Usage

#### Full Update Mode

Updates all fighters in the UFC database. This can take a while to complete:

```bash
python ssqlufc.py --mode full
```

#### Recent Fighters Mode

Only updates fighters who competed in recent events (much faster):

```bash
python ssqlufc.py --mode recent --events 2
```

Parameters:
- `--mode`: Choose between `full` or `recent` mode
- `--events`: Number of recent events to check (default: 2)

## UFC Fighter Last 5 Fights Scraper (l55.py)

This script scrapes and maintains the last 5 fights for each UFC fighter and stores them in an SQLite database.

### Features

- Scrapes detailed fight statistics for each fighter (strikes, takedowns, control time, etc.)
- Maintains up to 5 most recent fights per fighter
- When a fighter has a new fight, the oldest one is automatically removed
- Sorts fights by date (newest to oldest)
- Available in two operating modes: full database update or recent event only

### Usage

#### Full Update Mode

Updates all fighters in the UFC database with their last 5 fights. This will take a long time to complete:

```bash
python l55.py --mode full
```

#### Recent Event Mode

Only updates fighters who competed in the most recent event (much faster):

```bash
python l55.py --mode recent
```

#### Table Management

By default, the script recreates the `fighter_last_5_fights` table in full mode. If you want to just update existing records in full mode, you can use:

```bash
python l55.py --mode full --recreate False
```

## UFC Rankings Scraper (ufc_rankings_scraper.py)

This script fetches fighter rankings from the UFC website and stores them directly in the fighters table.

### Features
- Scrapes official UFC rankings for all weight classes
- Updates fighter records with current ranking information
- Identifies champions vs ranked contenders
- Supports debugging mode

### Usage
```bash
python ufc_rankings_scraper.py
```

## Tapology Scraper (tapscrap.py)

Optimized Tapology Scraper with anti-blocking measures and multithreading.

### Features
- Enhanced with adaptive throttling
- Comprehensive error handling
- Ability to run continuously
- Extracts additional fighter information not available on UFC Stats

### Usage
```bash
python tapscrap.py
```

## Database

All scrapers update the same SQLite database located at:
```
data/ufc_fighters.db
```

## Automatic Updates

For AWS Event Scheduler or cron jobs, use the recent mode to keep your database up-to-date efficiently:

```bash
# Update only fighters from the last 3 UFC events
python /path/to/ssqlufc.py --mode recent --events 3

# Update fight history for fighters from recent event
python /path/to/l55.py --mode recent

# Update rankings
python /path/to/ufc_rankings_scraper.py
```

## Notes

- The script preserves any custom fields you've added to the database
- Fields like `image_url`, `tap_link`, `ranking`, `is_champion`, etc. are maintained
- New fighters are added with default values for custom fields 