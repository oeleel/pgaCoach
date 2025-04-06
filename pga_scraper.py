import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import pickle
import os

def parse_numeric(text):
    """
    Attempt to convert a text string into a float.
    If it contains commas or percentage signs, remove them.
    If conversion fails, return the original text.
    """
    t = text.strip().replace("%", "").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return text.strip()

def parse_supporting(text):
    """
    For cells that contain a label and a value in parentheses,
    extract the label and the numeric value.
    For example, "Total SG:T (-7.119)" becomes ("Total SG:T", -7.119).
    If no match is found, return (text, None).
    """
    t = text.strip()
    match = re.search(r"^(.*?)\s*\(\s*([-+]?[0-9.,]+)\s*\)$", t)
    if match:
        label = match.group(1).strip()
        num = match.group(2).replace(",", "")
        try:
            value = float(num)
        except ValueError:
            value = num
        return label, value
    return t, None

# --- Define the Allowed Stats (exactly as provided) ---
allowed_stats = [
    "SG: Total", "SG: Tee-to-Green", "SG: Off-the-Tee", "SG: Approach the Green", "SG: Around-the-Green",
    "SG: Putting", "Longest Drives", "Driving Distance", "Driving Distance - All Drives", "Driving Accuracy Percentage",
    "Left Rough Tendency", "Right Rough Tendency", "Distance from Edge of Fairway", "Club Head Speed", "Total Driving Efficiency",
    "Greens in Regulation Percentage", "Proximity to Hole", "Approaches from > 275 yards", "Approaches from 250-275 yards",
    "Approaches from 225-250 yards", "Approaches from 200-225 yards", "Approaches from > 200 yards", "Approaches from 175-200 yards",
    "Approaches from 150-175 yards", "Approaches from 125-150 yards", "Approaches from 50-125 yards", "Approaches from 100-125 yards",
    "Approaches from 75-100 yards", "Approaches from 50-75 yards", "Approaches from inside 100 yards", "Approaches from > 100 yards",
    "Fairway Proximity", "Rough Proximity", "Left Rough Proximity", "Right Rough Proximity",
    "Approaches from > 275 yards (Rgh)", "Approaches from 250-275 yards (Rgh)", "Approaches from 225-250 yards (Rgh)",
    "Approaches from 200-225 yards (Rgh)", "Approaches from > 100 yards (Rgh)", "Approaches from inside 100 yards (Rgh)",
    "Approaches from > 200 yards (Rgh)", "Approaches from 175-200 yards (Rgh)", "Approaches from 150-175 yards (Rgh)",
    "Approaches from 125-150 yards (Rgh)", "Approaches from 50-125 yards (Rgh)", "Approaches from 100-125 yards (Rgh)",
    "Approaches from 75-100 yards (Rgh)", "Approaches from 50-75 yards (Rgh)", "Going for the Green",
    "Going for the Green - Hit Green Pct.", "Going for the Green - Birdie or Better", "Total Hole Outs",
    "Longest Hole Outs (in yards)", "Scrambling", "Scrambling from the Rough", "Scrambling from the Fringe",
    "Scrambling from > 30 yards", "Scrambling from 20-30 yards", "Scrambling from 10-20 yards", "Scrambling from < 10 yards",
    "Sand Save Percentage", "Proximity to Hole from Sand", "Total Putting", "Putting Average", "Overall Putting Average",
    "Birdie or Better Conversion Percentage", "Putts per Round", "Putts per Round - Round 1", "Putts per Round - Round 2",
    "Putts per Round - Round 3", "Putts per Round - Round 4", "One-Putt Percentage", "Longest Putts", "3-Putt Avoidance",
    "Putting from - > 25'", "Putting from - 20-25'", "Putting from - 15-20'", "Putting from - 10-15'", "Putting - Inside 10'",
    "Putting from 10'", "Putting from 9'", "Putting from 8'", "Putting from 7'", "Putting from 6'", "Putting from 5'",
    "Putting from 4-8'", "Putting from 4'", "Putting from 3'", "Average Distance of Putts made", "Approach Putt Performance",
    "Scoring Average (Adjusted)", "Scoring Average (Actual)", "Lowest Round", "Birdie Average", "Total Birdies",
    "Eagles (Holes per)", "Total Eagles", "Par Breakers", "Bounce Back", "Par 3 Birdie or Better Leaders",
    "Par 4 Birdie or Better Leaders", "Par 5 Birdie or Better Leaders", "Birdie or Better Percentage", "Bogey Avoidance",
    "Final Scoring Average", "Final Round Performance", "Round 1 Scoring Average", "Round 2 Scoring Average",
    "Round 3 Scoring Average", "Round 4 Scoring Average", "Par 3 Scoring Average", "Par 4 Scoring Average", "Par 5 Scoring Average",
    "Front 9 Scoring Average", "Back 9 Scoring Average", "Early Scoring Average", "Late Scoring Average", "Consecutive Cuts",
    "Current Streak with a 3-Putt", "Consecutive Fairways Hit", "Consecutive GIR", "Consecutive Sand Saves",
    "Best YTD 1-Putt or Better Streak", "Best YTD Streak w/o a 3-Putt", "YTD Par or Better Streak",
    "Consecutive Par 3 Birdies", "Consecutive Holes Below Par", "Consecutive Birdies Streak", "Consecutive Birdies/Eagles streak",
    "Official Money", "Comcast Business TOUR TOP 10"
]

# Create a set of allowed stats in lower case for faster lookup.
allowed_stats_set = set(stat.lower() for stat in allowed_stats)

def create_driver():
    """Create a new Chrome WebDriver instance"""
    options = Options()
    # Uncomment the next line to run in headless mode (without opening a browser window)
    # options.add_argument("--headless")
    return webdriver.Chrome(options=options)

def save_progress(players_data, processed_urls, output_filename="pgatour_players_stats.xlsx"):
    """Save the current progress to Excel and pickle files"""
    df = pd.DataFrame(players_data)
    df.to_excel(output_filename, index=False)
    with open("progress.pkl", "wb") as f:
        pickle.dump(processed_urls, f)
    print(f"Progress saved: {len(players_data)} players processed")

def load_progress():
    """Load previous progress if it exists"""
    players_data = []
    processed_urls = set()
    if os.path.exists("progress.pkl"):
        with open("progress.pkl", "rb") as f:
            processed_urls = pickle.load(f)
        if os.path.exists("pgatour_players_stats.xlsx"):
            df = pd.read_excel("pgatour_players_stats.xlsx")
            players_data = df.to_dict('records')
            print(f"Loaded previous progress: {len(players_data)} players")
    return players_data, processed_urls

# --- Setup Selenium WebDriver ---
driver = create_driver()

# --- Step 1: Scrape All Player Profile URLs ---
players_page_url = "https://www.pgatour.com/players"
print(f"Accessing players page: {players_page_url}")
driver.get(players_page_url)
time.sleep(5)

# Scroll to load all players
print("Starting to scroll to load all players...")
last_height = driver.execute_script("return document.body.scrollHeight")
scroll_count = 0
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count += 1
    print(f"Scroll {scroll_count}: Height changed from {last_height} to {new_height}")
    if new_height == last_height:
        break
    last_height = new_height

# Find player profile links (exclude those already containing '/stats')
print("Searching for player profile links...")
player_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/player/']")
player_urls = set()
for elem in player_elements:
    href = elem.get_attribute("href")
    if "/stats" in href:
        continue
    player_urls.add(href)

player_urls = list(player_urls)
print(f"Found {len(player_urls)} player profiles.")
if len(player_urls) == 0:
    print("WARNING: No player URLs found! This might indicate a problem with the page structure or loading.")
    driver.quit()
    exit(1)

# Load previous progress if it exists
players_data, processed_urls = load_progress()
remaining_urls = [url for url in player_urls if url not in processed_urls]
print(f"Starting scraping of {len(remaining_urls)} remaining players...")

# --- Step 2: Scrape Stats for Each Player and Aggregate into One Row per Player ---
start_time = time.time()
max_retries = 3
retry_delay = 5

for idx, profile_url in enumerate(remaining_urls, start=1):
    retry_count = 0
    while retry_count < max_retries:
        try:
            elapsed_time = time.time() - start_time
            avg_time_per_player = elapsed_time / (idx - 1) if idx > 1 else 0
            remaining_players = len(remaining_urls) - idx
            estimated_remaining_time = remaining_players * avg_time_per_player if avg_time_per_player > 0 else 0
            
            print(f"\nProcessing player {idx}/{len(remaining_urls)}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds")
            
            parts = profile_url.split("/")
            player_id = parts[parts.index("player") + 1] if "player" in parts else None
            player_name = profile_url.rstrip("/").split("/")[-1].replace("-", " ").title()
            stats_page_url = profile_url.rstrip("/") + "/stats"
            
            print(f"Accessing stats page for {player_name}: {stats_page_url}")
            driver.get(stats_page_url)
            time.sleep(5)
            
            # Scroll to load the table if needed
            print("Scrolling to load stats table...")
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # Locate the stats table
            print("Looking for stats table...")
            try:
                # Wait for table to be present
                table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table.chakra-table"))
                )
                rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
                print(f"Found {len(rows)} stat rows")
            except TimeoutException:
                print(f"WARNING: Timeout waiting for stats table for {player_name}")
                break
            except NoSuchElementException:
                print(f"WARNING: Could not find stats table for {player_name}")
                break

            if not rows:
                print(f"No stat rows found for: {profile_url}")
                break

            # Initialize player's data dictionary with metadata
            player_dict = {
                "Player ID": player_id,
                "Player": player_name,
                "Profile URL": profile_url,
                "Stats Page URL": stats_page_url
            }
            
            # Process each row and add only if the stat is in allowed_stats_set
            stats_collected = 0
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 5:
                        continue
                    stat_label = cells[0].text.strip()
                    # Check if this stat is one we want (case-insensitive)
                    if stat_label.lower() not in allowed_stats_set:
                        continue

                    primary_value = parse_numeric(cells[1].text.strip())
                    rank = parse_numeric(cells[2].text.strip())
                    supp_text = cells[3].text.strip()
                    supp_label, supp_value = parse_supporting(supp_text)
                    supp2_value = parse_numeric(cells[4].text.strip())
                    supporting = supp_value if supp_value is not None else supp2_value

                    key_primary = f"{stat_label} - Primary"
                    key_rank = f"{stat_label} - Rank"
                    key_supporting = f"{stat_label} - Supporting"
                    player_dict[key_primary] = primary_value
                    player_dict[key_rank] = rank
                    player_dict[key_supporting] = supporting
                    stats_collected += 1
                except Exception as e:
                    print(f"Error processing row for {player_name}: {str(e)}")
                    continue

            print(f"Collected {stats_collected} stats for {player_name}")
            if stats_collected > 0:
                players_data.append(player_dict)
                processed_urls.add(profile_url)
                print(f"Successfully added data for: {player_name} (ID: {player_id})")
                
                # Save progress every 10 players
                if len(players_data) % 10 == 0:
                    print(f"\nSaving progress... ({len(players_data)} players processed)")
                    save_progress(players_data, processed_urls)
            else:
                print(f"WARNING: No stats collected for {player_name}")
            break  # Successfully processed player, exit retry loop
        except WebDriverException as e:
            retry_count += 1
            print(f"Error processing {profile_url}: {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying in {retry_delay} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(retry_delay)
                try:
                    driver.quit()
                except:
                    pass
                driver = create_driver()
            else:
                print(f"Max retries reached for {profile_url}. Moving to next player.")
                break
        except Exception as e:
            print(f"Unexpected error processing {profile_url}: {str(e)}")
            break

# --- Step 3: Save Final Data to Excel File ---
print(f"\nSaving final data for {len(players_data)} players to Excel...")
if len(players_data) == 0:
    print("WARNING: No data collected for any players!")
    driver.quit()
    exit(1)

save_progress(players_data, processed_urls)
print(f"Total rows in DataFrame: {len(players_data)}")
print(f"Total columns in DataFrame: {len(players_data[0]) if players_data else 0}")
print(f"Total time taken: {time.time() - start_time:.2f} seconds")

driver.quit()