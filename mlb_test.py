import requests

# 1. Define the date range for the 2026 MLB Season (Spring through Fall)
start_date = "2026-03-01"
end_date = "2026-11-01"

print(f"Gathering full MLB season schedule from {start_date} to {end_date}...\n")

# 2. Ping the API with our broad date filters
url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
response = requests.get(url).json()

game_counter = 0

# 3. Step through every date block returned by the league server
if "dates" in response:
    for date_block in response["dates"]:
        date_string = date_block["date"] # The calendar date
        games = date_block["games"]
        
        for game in games:
            away_team = game["teams"]["away"]["team"]["name"]
            home_team = game["teams"]["home"]["team"]["name"]
            game_id = game["gamePk"]
            
            # Print the first few matches so we don't spam the terminal with 2,430 lines
            if game_counter < 15:
                print(f"📅 {date_string} | ⚾ {away_team} @ {home_team} (ID: {game_id})")
            
            game_counter += 1

    print("-" * 50)
    print(f"✅ Success! Loaded a total of {game_counter} games for the entire season.")
else:
    print("Failed to pull the seasonal layout. Double check your internet link.")