import pandas as pd

# Load your files
pit_stops = pd.read_csv("pit_stops.csv")
races = pd.read_csv("races.csv")       # This links raceId ↔ circuitId
circuits = pd.read_csv("circuits.csv") # This contains the "alt" feature

# 1. Merge pit stops with races to get circuitId
ps_with_race = pit_stops.merge(
    races[["raceId", "circuitId"]],
    on="raceId",
    how="left"
)
print(ps_with_race)
# 2. Merge with circuits to get "alt"
ps_with_alt = ps_with_race.merge(
    circuits[["circuitId", "alt"]],
    on="circuitId",
    how="left"
)

# Final dataframe
print(ps_with_alt.head())

# Save if needed
ps_with_alt.to_csv("pit_stops_with_alt.csv", index=False)
