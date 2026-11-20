"""es_time_prep_ext.py: preparation and pre-run checks of the extension datasets (separately registered
second look at early_stopping_time_plan.md). Imports the shared helpers of es_time_prep.py and applies the
same rules: time column, sort with a seeded random tie order, thin variant (10,000 random rows, seed 0,
indices sorted), the five pre-run checks with the original thresholds, y = 1 is the minority class. No
dense variant. No calendar column is derived except the hour of day. A categorical level with fewer than
RARE rows in the full prepared data is recoded to "other" in the columns named in each builder. The
single-feature screens of check 5 are also run on cuts 2 and 3 (_features_allcuts_*.csv).

Usage: python3 -I es_time_prep_ext.py <data_dir> <out_dir> <published_dir> [dataset ...]

<published_dir> is the prepared folder of the first run (es_time_prep.py). Before anything is written, the script
asserts that its manifest and the prepared thin files of the eight published datasets still have the hashes the
first run recorded, so their stored fits are reused as they are.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import es_time_prep as base  # noqa: E402

RARE = 100


def rare(s, min_count=RARE):
    """Recode levels with fewer than min_count rows to "other"; missing stays missing."""
    s = s.astype("string")
    counts = s.value_counts()
    keep = counts[counts >= min_count].index
    return s.where(s.isin(keep) | s.isna(), "other").astype(object).where(s.notna())


def sample_dvsa(raw, dest, p=0.02, seed=0):
    """Seeded Bernoulli sample (share p) of the normal tests (test_type NT) with a pass or fail result in the
    DVSA 2023 results file. A uniform subset of 10,000 rows drawn from this sample is a uniform subset of the
    full file, so the thin variant keeps its definition; only the full-data row count is reduced."""
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    for ch in pd.read_csv(raw, sep="|", chunksize=2_000_000, low_memory=False):
        ch = ch[(ch.test_type == "NT") & ch.test_result.isin(["P", "F"])]
        total += len(ch)
        parts.append(ch[rng.random(len(ch)) < p])
    pd.concat(parts).to_csv(dest, index=False)
    return total


def sample_tlc(folder, dest, p=0.01, seed=0):
    """Seeded Bernoulli sample (share p) of the credit-card trips (tips are recorded only for them) in the twelve
    2023 yellow taxi files, as for sample_dvsa. Returns the number of such trips in the files."""
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    for m in range(1, 13):
        f = pd.read_parquet(f"{folder}/yellow_tripdata_2023-{m:02d}.parquet")
        f = f[(f.payment_type == 1) & (f.fare_amount > 0) & (f.tpep_pickup_datetime.dt.year == 2023)]
        total += len(f)
        parts.append(f[rng.random(len(f)) < p])
    pd.concat(parts).to_csv(dest, index=False)
    return total


def sample_citi(folder, dest, p=0.01, seed=0):
    """Seeded Bernoulli sample (share p) of the rides in the twelve 2023 Citi Bike monthly archives (each holds
    several CSV parts). Returns the number of rides in the files."""
    import zipfile
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    for m in range(1, 13):
        with zipfile.ZipFile(f"{folder}/2023-citibike-tripdata/2023{m:02d}-citibike-tripdata.zip") as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")):
                with z.open(name) as fh:
                    for ch in pd.read_csv(fh, chunksize=1_000_000, low_memory=False):
                        total += len(ch)
                        parts.append(ch[rng.random(len(ch)) < p])
    pd.concat(parts).to_csv(dest, index=False)
    return total


CFPB_FINAL = ["Closed with explanation", "Closed with non-monetary relief", "Closed with monetary relief",
              "Closed without relief", "Closed with relief", "Closed"]


def sample_cfpb(raw, dest, p=0.01, seed=0):
    """Seeded Bernoulli sample (share p) of the CFPB complaints with a final company response (rows still
    "In progress" or "Untimely response" are dropped). As for DVSA, the thin variant keeps its definition."""
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    for ch in pd.read_csv(raw, chunksize=1_000_000, low_memory=False, compression="zip",
                          usecols=["Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Company", "State",
                                   "Tags", "Submitted via", "Date sent to company", "Company response to consumer"]):
        ch = ch[ch["Company response to consumer"].isin(CFPB_FINAL)]
        total += len(ch)
        parts.append(ch[rng.random(len(ch)) < p])
    pd.concat(parts).to_csv(dest, index=False)
    return total


LICHESS_TAGS = ("UTCDate", "UTCTime", "Result", "WhiteElo", "BlackElo", "TimeControl", "Event")


def sample_lichess(folder, dest, p=0.01, seed=0):
    """Seeded Bernoulli sample (share p) of the games in the twelve 2013 Lichess monthly rated-game files, as
    for sample_dvsa. Only the header tags in LICHESS_TAGS are read; games with a result other than 1-0, 0-1 or
    1/2-1/2, or without both ratings, are not counted. Returns the number of counted games."""
    import io
    import subprocess
    rng = np.random.default_rng(seed)
    rows, total = [], 0
    for m in range(1, 13):
        f = f"{folder}/lichess_db_standard_rated_2013-{m:02d}.pgn.zst"
        game = {}
        for line in io.TextIOWrapper(subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE).stdout,
                                     errors="replace"):
            if line.startswith("[") and " " in line:
                tag, _, val = line[1:].partition(" ")
                if tag in LICHESS_TAGS:
                    game[tag] = val.strip().strip("]").strip('"')
            elif line.startswith("1.") or (line.strip() and not line.startswith("[")):
                if len(game) == len(LICHESS_TAGS) and game["Result"] in ("1-0", "0-1", "1/2-1/2"):
                    total += 1
                    if rng.random() < p:
                        rows.append(game)
                game = {}
    pd.DataFrame(rows).to_csv(dest, index=False)
    return total


def prep(data):
    inputs, out = {}, {}

    def read(rel, **kw):
        inputs[rel] = base.sha256(f"{data}/{rel}")
        return pd.read_csv(f"{data}/{rel}", **kw)

    # UCI occupancy detection (Candanedo and Feldheim 2016): the three published files, one room, one
    # reading per minute. Ground truth is the occupancy at the reading. No column dropped.
    parts = []
    for f in ["datatraining.txt", "datatest.txt", "datatest2.txt"]:
        parts.append(read(f"occ/{f}"))
    o = pd.concat(parts, ignore_index=True)
    o["t"] = pd.to_datetime(o.date)
    o["y"] = o.Occupancy
    out["occupancy"] = (o.drop(columns=["date", "Occupancy"]), "occupied", "empty")

    # Austin Animal Center outcomes. Time = outcome date (local wall time, the first 19 characters, as some rows carry no offset). y = adopted. Dropped: animal id and name,
    # date of birth (absolute time; age at outcome kept), month-year (calendar),
    # outcome type and subtype (the label), and the neutered or intact status in "Sex upon Outcome" (the
    # shelter alters animals before adoption, so it is set by the outcome process; sex alone is kept). The
    # name flag is dropped by the same rule (the shelter names animals during the stay, a process that
    # prepares the outcome).
    a = read("austin_outcomes.csv")
    a["t"] = pd.to_datetime(a.DateTime.str[:19])
    a["y"] = (a["Outcome Type"] == "Adoption").astype(int)
    num = a["Age upon Outcome"].str.extract(r"(\d+)\s+(\w+?)s?$")
    unit = num[1].map({"day": 1, "week": 7, "month": 30, "year": 365})
    a["age_days"] = pd.to_numeric(num[0]) * unit
    a["sex"] = a["Sex upon Outcome"].str.extract(r"(Male|Female)")[0].fillna("Unknown")
    a["Breed"], a["Color"] = rare(a.Breed), rare(a.Color)
    out["austin"] = (a[["t", "y", "Animal Type", "sex", "age_days", "Breed", "Color"]],
                     "adopted", "other outcome")

    # Seattle Terry stops. Time = occurred date. y = Arrest Flag, whose first arrest is on 2019-05-08: no arrest
    # in the 34,159 source rows before that date, 23 % of the rows from it are arrests, so rows before
    # 2019-05-08 are dropped. The column is the label because the stop resolution, which does record
    # arrests earlier, is dropped as the label's record. Dropped columns: stop resolution, weapon type and
    # final call type (fixed after the stop), frisk flag (an action taken during the stop), identifiers
    # (subject id, report numbers, stop id, officer id), reported date (filing lag). The officer's year of
    # birth is kept as the officer's age at the stop (absolute time removed, as for the Austin animals).
    s = read("seattle_stops.csv")
    s["t"] = pd.to_datetime(s["Occurred Date"].str[:19])
    s = s[s.t.notna() & (s.t >= "2019-05-08")].copy()
    s["y"] = (s["Arrest Flag"] == "Y").astype(int)
    s["officer_age"] = s.t.dt.year - pd.to_numeric(s["Officer YOB"], errors="coerce").where(lambda v: v > 1900)
    for c in ["Initial Call Type", "Officer Squad", "Beat", "Sector", "Precinct"]:
        s[c] = rare(s[c])
    out["seattle-stops"] = (s[["t", "y", "Subject Age Group", "officer_age", "Officer Gender", "Officer Race",
                               "Subject Perceived Race", "Subject Perceived Gender", "Initial Call Type",
                               "Call Type", "Officer Squad", "Precinct", "Sector", "Beat"]],
                            "arrest", "no arrest")

    # NYPD shooting incidents (historic). Time = occurrence date and time. y = statistical murder flag
    # (the victim died). Dropped: the perpetrator columns (filled in as the investigation proceeds),
    # incident key (sequential), projected X and Y (duplicates of latitude and longitude), the
    # longitude-latitude text, the date (time of day is kept as hour), and the three location-description
    # columns LOC_OF_OCCUR_DESC, LOC_CLASSFCTN_DESC and LOCATION_DESC (the first two
    # missing in every row before 2022, the third in 59 %, all filled afterwards; drift AUC 0.99, 0.98 and
    # 0.92 on cut 3; the first plan dropped bank-full contact for the same reason).
    n = read("nypd_shooting.csv")
    n["t"] = pd.to_datetime(n.OCCUR_DATE + " " + n.OCCUR_TIME, format="%m/%d/%Y %H:%M:%S")
    n["y"] = n.STATISTICAL_MURDER_FLAG.astype(int)
    n["label_known"] = n.t + pd.Timedelta(days=30)  # assumed: the flag can change when a victim dies later; the source gives no date
    n["hour"] = n.t.dt.hour
    n["PRECINCT"] = n.PRECINCT.astype(str)
    n["JURISDICTION_CODE"] = n.JURISDICTION_CODE.astype("Int64").astype(str).where(n.JURISDICTION_CODE.notna())
    out["nypd-shooting"] = (n[["t", "y", "label_known", "BORO", "PRECINCT", "JURISDICTION_CODE", "VIC_AGE_GROUP", "VIC_SEX",
                               "VIC_RACE", "Latitude", "Longitude", "hour"]], "murder", "survived")

    # Chicago food inspections. Time = inspection date. y = Fail among Pass, Pass w/ Conditions and Fail
    # (other results are not an outcome of the inspection and are dropped). Dropped: the violations text
    # (the record of the outcome), inspection id, license number (sequential), names, address, city,
    # state and the location text (identifiers).
    c = pd.read_csv(f"{data}/chicago_food.csv", usecols=lambda col: col != "Violations", low_memory=False)
    inputs["chicago_food.csv"] = base.sha256(f"{data}/chicago_food.csv")
    c = c[c.Results.isin(["Pass", "Pass w/ Conditions", "Fail"])].copy()
    c["t"] = pd.to_datetime(c["Inspection Date"], format="%m/%d/%Y")
    c["y"] = (c.Results == "Fail").astype(int)
    c["Zip"] = c.Zip.astype("Int64").astype(str).where(c.Zip.notna())
    c["Facility Type"], c["Inspection Type"] = rare(c["Facility Type"]), rare(c["Inspection Type"])
    out["chicago-food"] = (c[["t", "y", "Facility Type", "Risk", "Zip", "Inspection Type", "Latitude",
                              "Longitude"]], "fail", "pass")
    # UK road safety data (STATS19), collisions of the last five published years (2021-2025). Time =
    # collision date and time. y = fatal or serious (severity 1 or 2) against slight; a later death does not change the label, so no label
    # date is modelled. Dropped: index,
    # year and reference number (identifiers), projected coordinates (duplicates of longitude and
    # latitude), day of week (calendar), the four local authority codes (the district code is -1 in 99.96 % of
    # rows, the others are identifier-like levels), the four columns ending _historic (older codings of
    # junction detail, pedestrian crossing and carriageway hazards, kept in the file beside the current
    # ones), the road number and
    # the LSOA code (identifier-like levels), the number of casualties (a consequence of the collision,
    # severity is the worst casualty's injury), whether police attended (recorded after), and the
    # derived severity columns (enhanced, injury-based, adjusted). Coded columns are categorical.
    c5 = read("stats19_collisions_5y.csv", low_memory=False)
    c5["t"] = pd.to_datetime(c5.date + " " + c5.time, format="%d/%m/%Y %H:%M")
    c5["y"] = (c5.collision_severity < 3).astype(int)
    c5["hour"] = c5.t.dt.hour
    coded = ["police_force", "first_road_class", "road_type", "junction_detail",
             "junction_control", "second_road_class", "pedestrian_crossing", "light_conditions",
             "weather_conditions", "road_surface_conditions", "special_conditions_at_site",
             "carriageway_hazards", "urban_or_rural_area", "trunk_road_flag"]
    for col in coded:
        c5[col] = c5[col].astype(str)
    out["stats19"] = (c5[["t", "y", "number_of_vehicles", "speed_limit", "longitude", "latitude", "hour"] + coded],
                      "fatal or serious", "slight")
    # London Fire Brigade incident records 2009-2017. Time = date and time of call. y = false alarm
    # (incident group). Dropped: incident number and year (identifiers, calendar), special service type
    # (filled only for special services, so it answers the label), stop code description (the label's
    # duplicate), every attendance, pump and cost column (outcomes of the response), number of calls (it
    # accumulates after the first call), postcode, UPRN and USRN (identifiers), ward-name variants,
    # borough duplicates, projected coordinates.
    lf = read("lfb_2009_2017.csv", low_memory=False)
    lf = lf[lf.IncidentGroup.notna()].copy()
    lf["t"] = pd.to_datetime(lf.DateOfCall + " " + lf.TimeOfCall, format="%d-%b-%y %H:%M:%S")
    lf["y"] = (lf.IncidentGroup == "False Alarm").astype(int)
    for col in ["PropertyType", "IncidentStationGround", "IncGeo_WardName", "Postcode_district"]:
        lf[col] = rare(lf[col])
    out["london-fire"] = (lf[["t", "y", "HourOfCall", "PropertyCategory", "PropertyType", "AddressQualifier",
                              "IncGeo_BoroughName", "IncGeo_WardName", "Postcode_district",
                              "IncidentStationGround", "Latitude", "Longitude"]], "false alarm", "real incident")
    # DVSA anonymised MOT tests, results file 2023 (test dates 2023-01-01 to 2023-12-31). Normal tests (NT)
    # with result pass or fail; retests and partial results are dropped. y = fail. The file holds 32.8 million
    # such tests; a seeded 2 % Bernoulli sample (sample_dvsa) stands in for the full file.
    # Time = test date. Dropped: test and vehicle ids (identifiers), test type (constant), the first use
    # date (absolute time; the vehicle age at test in years is kept). Make and model levels under RARE
    # rows in the sample are "other".
    dv_path = Path(data) / "dvsa_2023_sample.csv"
    if not dv_path.exists():
        total = sample_dvsa(f"{data}/dvsa/test_result.csv", dv_path)
        print(f"dvsa: {total} normal pass/fail tests in the file, sample written")
    inputs["dvsa/test_result.csv"] = base.sha256(f"{data}/dvsa/test_result.csv")
    dv = pd.read_csv(dv_path, low_memory=False)
    dv["t"] = pd.to_datetime(dv.test_date)
    dv["y"] = (dv.test_result == "F").astype(int)
    dv["age_years"] = (dv.t - pd.to_datetime(dv.first_use_date, errors="coerce")).dt.days / 365.25
    dv["test_class_id"] = dv.test_class_id.astype(str)
    for col in ["make", "model", "colour", "postcode_area"]:
        dv[col] = rare(dv[col])
    out["dvsa-mot"] = (dv[["t", "y", "test_class_id", "test_mileage", "postcode_area", "make", "model",
                           "colour", "fuel_type", "cylinder_capacity", "age_years"]], "fail", "pass")
    # NYC motor vehicle collisions (crashes). Time = crash date and time. y = at least one person injured or
    # killed. Rows whose date does not parse (a few malformed lines) are dropped. Dropped: collision id,
    # location text and street names (identifiers), every injury and death count (the label), the
    # contributing factors (judged by the police after the event), vehicle types 3 to 5 (rarely filled).
    # Vehicle types 1 and 2 stay with levels under RARE rows recoded to "other".
    co = read("collisions.csv", low_memory=False)
    co["t"] = pd.to_datetime(co["CRASH DATE"], format="%m/%d/%Y", errors="coerce")
    co = co[co.t.notna()].copy()
    co["t"] = co.t + pd.to_timedelta(pd.to_datetime(co["CRASH TIME"], format="%H:%M", errors="coerce").dt.hour.fillna(0), unit="h")
    co["y"] = ((pd.to_numeric(co["NUMBER OF PERSONS INJURED"], errors="coerce").fillna(0)
                + pd.to_numeric(co["NUMBER OF PERSONS KILLED"], errors="coerce").fillna(0)) > 0).astype(int)
    co["hour"] = co.t.dt.hour
    co["ZIP CODE"] = co["ZIP CODE"].astype(str).where(co["ZIP CODE"].notna())
    for col in ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2"]:
        co[col] = rare(co[col])
    out["nyc-collisions"] = (co[["t", "y", "BOROUGH", "ZIP CODE", "LATITUDE", "LONGITUDE", "hour",
                                 "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2"]], "injury or death", "none")
    # NYC TLC yellow taxi trips 2023 (twelve monthly files). Credit-card trips only (cash tips are not
    # recorded), fare above zero. y = no tip. The trips are sampled as in sample_tlc (1 %). Time = pickup.
    # Dropped: drop-off time (absolute time), tip and total amount (the label, and the total contains
    # it), payment type (constant). Kept although known only at the end of the trip: distance, fare,
    # tolls, extra charge and drop-off zone (the label, the tip, is determined at payment).
    tl_path = Path(data) / "tlc_2023_sample.csv"
    if not tl_path.exists():
        total = sample_tlc(f"{data}/tlc", tl_path)
        print(f"tlc: {total} credit-card trips in the files, sample written")
    for m in range(1, 13):
        inputs[f"tlc/yellow_tripdata_2023-{m:02d}.parquet"] = base.sha256(f"{data}/tlc/yellow_tripdata_2023-{m:02d}.parquet")
    tl = pd.read_csv(tl_path, low_memory=False)
    tl["t"] = pd.to_datetime(tl.tpep_pickup_datetime)
    tl["y"] = (tl.tip_amount <= 0).astype(int)
    tl["hour"] = tl.t.dt.hour
    for col in ["VendorID", "RatecodeID", "PULocationID", "DOLocationID", "store_and_fwd_flag"]:
        tl[col] = tl[col].astype("string").astype(object).where(tl[col].notna())
    out["tlc-yellow"] = (tl[["t", "y", "VendorID", "passenger_count", "trip_distance", "RatecodeID",
                             "store_and_fwd_flag", "PULocationID", "DOLocationID", "fare_amount", "extra",
                             "mta_tax", "tolls_amount", "improvement_surcharge", "congestion_surcharge",
                             "Airport_fee", "hour"]], "no tip", "tip")
    # SFPD incident reports, historical 2003 to May 2018. Time = date and time of the incident. y =
    # resolution is an arrest (booked or cited). Dropped: record and incident numbers (sequential),
    # incident code (duplicate of the description), day of week (calendar), address, the location text, the
    # boundary-lookup columns joined in later, load date, and the resolution (the label). Description
    # levels under RARE rows are "other".
    sf = read("sfpd_hist.csv", usecols=["Category", "Descript", "Date", "Time", "PdDistrict", "Resolution", "X", "Y"],
              low_memory=False)
    sf = sf[sf.Resolution.notna()].copy()
    sf["t"] = pd.to_datetime(sf.Date + " " + sf.Time, format="%m/%d/%Y %H:%M")
    sf["y"] = sf.Resolution.str.startswith("ARREST").astype(int)
    sf["hour"] = sf.t.dt.hour
    sf["Descript"] = rare(sf.Descript)
    out["sfpd"] = (sf[["t", "y", "Category", "Descript", "PdDistrict", "X", "Y", "hour"]], "arrest", "no arrest")

    # nflverse play-by-play, seasons 2015-2024. Time = game date (many plays per date, ties broken at
    # random). Rows are scrimmage plays that were a pass or a run. y = run (the minority). Features are
    # known before the snap. Dropped: ids, the description, week and season (calendar), every outcome of the
    # play (yards, result fields), the quarterback dropback flag (it records what the quarterback did after
    # the snap and nearly copies the label), and every output of a fitted model (expected points, win
    # probability), because those models were fitted on seasons that include later test blocks, as for the
    # expected-goal columns of the NHL shots. The pre-game betting lines are kept.
    nfl_cols = ["game_date", "play_type", "posteam", "defteam", "posteam_type", "season_type", "yardline_100", "qtr",
                "down", "goal_to_go", "ydstogo", "game_seconds_remaining", "half_seconds_remaining",
                "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "shotgun",
                "no_huddle", "spread_line", "total_line", "roof", "surface", "temp", "wind"]
    nf = pd.concat([read(f"nfl/play_by_play_{y}.csv.gz", usecols=nfl_cols, low_memory=False)
                    for y in range(2015, 2025)], ignore_index=True)
    nf = nf[nf.play_type.isin(["pass", "run"]) & nf.down.notna()].copy()
    nf["t"] = pd.to_datetime(nf.game_date)
    nf["y"] = (nf.play_type == "run").astype(int)
    for col in ["qtr", "down"]:
        nf[col] = nf[col].astype(int).astype(str)
    out["nflverse"] = (nf.drop(columns=["game_date", "play_type"]), "run", "pass")
    # MoneyPuck NHL shot data, seasons 2020-2024. Time = season and game id (ids run in game order within a
    # season; the files carry no date, ties within a game broken at random). y = goal. Only pre-shot
    # columns are kept (a whitelist): shot type, geometry, period and clock, skaters and empty-net state, the
    # last event and its distance, speed and time, rush and rebound flags, shooter hand, position and ice
    # time, zone, team codes. Dropped: ids, names, the score columns, every post-shot field (on goal,
    # rebound generated, goalie froze, play continued, play stopped, time until next event), all expected
    # goal model outputs, and the event type (it contains the label).
    mp_cols = ["season", "game_id", "goal", "shotType", "shotDistance", "shotAngle", "shotAngleAdjusted",
               "xCordAdjusted", "yCordAdjusted", "period", "time", "isHomeTeam", "isPlayoffGame",
               "homeSkatersOnIce", "awaySkatersOnIce", "homeEmptyNet", "awayEmptyNet", "shotRush", "shotRebound",
               "lastEventCategory", "distanceFromLastEvent", "timeSinceLastEvent", "speedFromLastEvent",
               "shooterLeftRight", "playerPositionThatDidEvent", "offWing", "timeSinceFaceoff",
               "shooterTimeOnIce", "location", "homeTeamCode", "awayTeamCode"]
    mp = pd.concat([read(f"mp/shots_{y}.csv", usecols=mp_cols, low_memory=False) for y in range(2020, 2025)],
                   ignore_index=True)
    mp["t"] = mp.season * 100_000 + mp.game_id
    mp["y"] = mp.goal.astype(int)
    out["moneypuck"] = (mp.drop(columns=["season", "game_id", "goal"]), "goal", "no goal")
    # Citi Bike trip histories 2023 (twelve monthly archives), sampled as in sample_citi (1 %). Time = ride
    # start. y = casual rider (against member), a status held before the ride. Dropped: ride id, end time,
    # end station, end coordinates (known only at the end of the ride), station names (the ids stay), and
    # rides without a start station id or coordinates. Station id levels under RARE rows are "other".
    cb_path = Path(data) / "citibike_2023_sample.csv"
    if not cb_path.exists():
        total = sample_citi(f"{data}/citibike", cb_path)
        print(f"citibike: {total} rides in the files, sample written")
    for m in range(1, 13):
        inputs[f"citibike/2023-citibike-tripdata/2023{m:02d}-citibike-tripdata.zip"] = base.sha256(
            f"{data}/citibike/2023-citibike-tripdata/2023{m:02d}-citibike-tripdata.zip")
    cb = pd.read_csv(cb_path, low_memory=False)
    cb = cb[cb.start_station_id.notna() & cb.start_lat.notna() & cb.member_casual.isin(["member", "casual"])].copy()
    cb["t"] = pd.to_datetime(cb.started_at)
    cb["y"] = (cb.member_casual == "casual").astype(int)
    cb["hour"] = cb.t.dt.hour
    cb["start_station_id"] = rare(cb.start_station_id.astype(str))
    out["citibike"] = (cb[["t", "y", "rideable_type", "start_station_id", "start_lat", "start_lng", "hour"]],
                       "casual rider", "member")

    # Medical appointment no-shows (Brazil, Kaggle joniarroba). Time = the moment the appointment was booked
    # (ScheduledDay, has a clock time; the appointment day is a date only), label known at the end of the
    # appointment day. y = no-show. Dropped: patient and appointment ids (the appointment id grows with the
    # booking time, so it is a clock), the appointment date (calendar; its distance from the booking is kept as
    # lead_days, known when booking). Rows with a negative age or an appointment before its booking day are
    # data errors and dropped.
    ns = read("kg_noshowappointments/KaggleV2-May-2016.csv")
    ns["t"] = pd.to_datetime(ns.ScheduledDay).dt.tz_localize(None)
    ap = pd.to_datetime(ns.AppointmentDay).dt.tz_localize(None)
    ns["lead_days"] = (ap - ns.t.dt.normalize()).dt.days
    ns = ns[(ns.Age >= 0) & (ns.lead_days >= 0)].copy()
    ns["label_known"] = ap[ns.index] + pd.Timedelta(days=1)
    ns["y"] = (ns["No-show"] == "Yes").astype(int)
    ns["Neighbourhood"] = rare(ns.Neighbourhood)
    out["no-shows"] = (ns[["t", "y", "label_known", "Gender", "Age", "Neighbourhood", "Scholarship", "Hipertension",
                           "Diabetes", "Alcoholism", "Handcap", "SMS_received", "lead_days"]],
                       "no-show", "attended")

    # Kickstarter projects (Kaggle kemical, 2018-01 snapshot). Time = deadline (the order given in the
    # candidate list), which is when the label is known. y = successful, among projects that ended
    # successful or failed (canceled, live, undefined and suspended dropped). Dropped: id and name, the
    # funding columns pledged, usd pledged, usd_pledged_real and backers (the outcome), the goal in local
    # currency (usd_goal_real kept), the launch date (absolute time; kept as duration_days). Launch dates
    # before 2009 are placeholders and their rows are dropped.
    k = read("kg_kickstarter-projects/ks-projects-201801.csv")
    launched = pd.to_datetime(k.launched)
    k["t"] = pd.to_datetime(k.deadline)
    k = k[k.state.isin(["successful", "failed"]) & (launched >= "2009-01-01")].copy()
    k["duration_days"] = (k.t - launched[k.index].dt.normalize()).dt.days
    k["y"] = (k.state == "successful").astype(int)
    k["category"] = rare(k.category)
    out["kickstarter"] = (k[["t", "y", "category", "main_category", "currency", "usd_goal_real",
                             "country", "duration_days"]], "successful", "failed")

    # Olist orders (Brazil, Kaggle olistbr, 2016-2018). Time = purchase, label known at delivery. y = delivered
    # after the estimated delivery date, among delivered orders. Dropped: order and customer ids, review data
    # (after delivery), approval, carrier hand-off and delivery timestamps (after the purchase; the estimated
    # delivery and the seller's shipping limit are set at purchase and kept as days from the purchase), zip
    # prefixes and city names (high-cardinality geography; the state is kept). Items are aggregated per order.
    og = "kg_brazilian-ecommerce/olist_"
    od = read(og + "orders_dataset.csv")
    od = od[(od.order_status == "delivered") & od.order_delivered_customer_date.notna()].copy()
    od["t"] = pd.to_datetime(od.order_purchase_timestamp)
    od["label_known"] = pd.to_datetime(od.order_delivered_customer_date)
    od["y"] = (od.label_known > pd.to_datetime(od.order_estimated_delivery_date)).astype(int)
    od["est_days"] = (pd.to_datetime(od.order_estimated_delivery_date) - od.t).dt.days
    it = read(og + "order_items_dataset.csv").merge(read(og + "products_dataset.csv"), on="product_id", how="left") \
        .merge(read(og + "sellers_dataset.csv"), on="seller_id", how="left")
    ag = it.groupby("order_id").agg(n_items=("order_item_id", "count"), price=("price", "sum"),
                                    freight=("freight_value", "sum"), weight_g=("product_weight_g", "sum"),
                                    category=("product_category_name", "first"), seller_state=("seller_state", "first"),
                                    n_sellers=("seller_id", "nunique"), limit=("shipping_limit_date", "min"))
    pay = read(og + "order_payments_dataset.csv").groupby("order_id").agg(
        payment_type=("payment_type", "first"), installments=("payment_installments", "max"))
    od = od.merge(ag, on="order_id").merge(pay, on="order_id", how="left") \
        .merge(read(og + "customers_dataset.csv")[["customer_id", "customer_state"]], on="customer_id")
    od["limit_days"] = (pd.to_datetime(od.limit) - od.t).dt.total_seconds() / 86400
    for c in ["category", "seller_state", "customer_state"]:
        od[c] = rare(od[c])
    out["olist"] = (od[["t", "y", "label_known", "est_days", "n_items", "price", "freight", "weight_g", "category",
                        "seller_state", "n_sellers", "payment_type", "installments", "customer_state", "limit_days"]],
                    "late delivery", "on time")

    # Dota 2 matches (Kaggle devinanzelmo, public matches 2015-2016). Time = match start, label known at its end
    # (start + duration). y = radiant win. Features known before the match: the ten hero picks, one column per
    # hero (+1 radiant, -1 dire, 0 not picked), the game mode and the server cluster. Dropped: duration,
    # first blood, tower and barracks status, votes (all from the played match), every per-player statistic
    # and the match id.
    dm = read("kg_dota-2-matches/match.csv")
    pl = read("kg_dota-2-matches/players.csv", usecols=["match_id", "hero_id", "player_slot"])
    pl = pl[pl.hero_id > 0]
    pl = pl.assign(side=np.where(pl.player_slot < 128, 1, -1))
    heroes = pl.pivot_table(index="match_id", columns="hero_id", values="side", aggfunc="sum", fill_value=0)
    heroes.columns = [f"hero_{c}" for c in heroes.columns]
    dm = dm.merge(heroes, left_on="match_id", right_index=True)
    dm["t"] = pd.to_datetime(dm.start_time, unit="s")
    dm["label_known"] = dm.t + pd.to_timedelta(dm.duration, unit="s")
    dm["y"] = dm.radiant_win.astype(int)
    dm["game_mode"], dm["cluster"] = rare(dm.game_mode.astype(str)), rare(dm.cluster.astype(str))
    out["dota2"] = (dm[["t", "y", "label_known", "game_mode", "cluster"] + list(heroes.columns)],
                    "radiant win", "dire win")

    # NYC 311 service requests created in 2023 (SODA API, requests whose unique_key ends in 00: a fixed 1 %
    # of ids, which carry no information about the request). Time = creation. y = closed within three days
    # (a horizon fixed before looking at the data; open requests count as not closed), label known three
    # days after creation. Dropped: unique_key, closed date (the label), and every field not listed in the
    # request (status, resolution text, due date, all fields set during handling). Agency, complaint type,
    # descriptor, location type and zip levels under RARE rows are "other". The monthly files were requested
    # from https://data.cityofnewyork.us/resource/erm2-nwe9.csv with $select=unique_key,created_date,closed_date,
    # agency,complaint_type,descriptor,location_type,incident_zip,borough,open_data_channel_type,latitude,
    # longitude and $where=created_date between '2023-MM-01T00:00:00' and '2023-MM-<last day>T23:59:59' and
    # unique_key like '%00' ($limit=50000).
    parts = []
    for m in range(1, 13):
        inputs[f"nyc311/2023-{m:02d}.csv"] = base.sha256(f"{data}/nyc311/2023-{m:02d}.csv")
        parts.append(pd.read_csv(f"{data}/nyc311/2023-{m:02d}.csv", low_memory=False))
    n3 = pd.concat(parts, ignore_index=True)
    n3["t"] = pd.to_datetime(n3.created_date)
    n3["label_known"] = n3.t + pd.Timedelta(days=3)
    n3["y"] = ((pd.to_datetime(n3.closed_date) - n3.t) <= pd.Timedelta(days=3)).astype(int)
    for c in ["complaint_type", "descriptor", "location_type", "incident_zip", "agency"]:
        n3[c] = rare(n3[c].astype(str).where(n3[c].notna()))
    out["nyc-311"] = (n3[["t", "y", "label_known", "agency", "complaint_type", "descriptor", "location_type",
                          "incident_zip", "borough", "open_data_channel_type", "latitude", "longitude"]],
                      "closed within 3 days", "slower or open")

    # Lichess rated games, 2013 (twelve monthly archives), sampled as in sample_lichess (1 %). Time = game
    # start (UTC date and time), label known at the end of the game (minutes, taken as the start). y = white
    # win (draws and black wins are 0). Dropped: player names, ids, site, termination, rating changes and
    # moves (all from the played game); the opening code and name (from the moves). Features known at the
    # pairing: both ratings, the time control and the event type.
    lc_path = Path(data) / "lichess_2013_sample.csv"
    if not lc_path.exists():
        total = sample_lichess(f"{data}/lichess", lc_path)
        print(f"lichess: {total} games in the files, sample written")
    for m in range(1, 13):
        inputs[f"lichess/lichess_db_standard_rated_2013-{m:02d}.pgn.zst"] = base.sha256(
            f"{data}/lichess/lichess_db_standard_rated_2013-{m:02d}.pgn.zst")
    lc = pd.read_csv(lc_path, dtype=str)
    for c in ["WhiteElo", "BlackElo"]:
        lc[c] = pd.to_numeric(lc[c], errors="coerce")
    lc = lc[lc.WhiteElo.notna() & lc.BlackElo.notna()].copy()
    lc["t"] = pd.to_datetime(lc.UTCDate.str.replace(".", "-", regex=False) + " " + lc.UTCTime)
    lc["y"] = (lc.Result == "1-0").astype(int)
    lc["TimeControl"], lc["Event"] = rare(lc.TimeControl), rare(lc.Event)
    out["lichess"] = (lc[["t", "y", "WhiteElo", "BlackElo", "TimeControl", "Event"]], "white win", "no white win")

    # CFPB Consumer Complaint Database (complaints.csv.zip, downloaded 2026-11-20), complaints with a final
    # company response, 1 % seeded sample (sample_cfpb). Time = date received. y = closed with monetary or
    # non-monetary relief. Label known 60 days after the date sent to the company (an assumption: the
    # response period the CFPB sets is 15 days, longer for complex cases). Dropped: complaint id, ZIP code
    # (high-cardinality; State kept), company public response and timely-response flag (set with the
    # response), date sent to company (after receipt). Company, sub-product, issue and sub-issue levels
    # under RARE rows are "other".
    cf_path = Path(data) / "cfpb_sample.csv"
    if not cf_path.exists():
        total = sample_cfpb(f"{data}/cfpb/complaints.csv.zip", cf_path)
        print(f"cfpb: {total} complaints with a final response, sample written")
    inputs["cfpb/complaints.csv.zip"] = base.sha256(f"{data}/cfpb/complaints.csv.zip")
    cf = pd.read_csv(cf_path)
    cf["t"] = pd.to_datetime(cf["Date received"])
    cf["label_known"] = pd.to_datetime(cf["Date sent to company"]) + pd.Timedelta(days=60)
    cf["y"] = cf["Company response to consumer"].isin(["Closed with monetary relief", "Closed with non-monetary relief",
                                                      "Closed with relief"]).astype(int)
    for c in ["Sub-product", "Issue", "Sub-issue", "Company"]:
        cf[c] = rare(cf[c])
    out["cfpb"] = (cf[["t", "y", "label_known", "Product", "Sub-product", "Issue", "Sub-issue", "Company", "State",
                       "Tags", "Submitted via"]], "relief", "no relief")
    return out, inputs


# sha256 of results/es_time/prepared/_manifest.json recorded in run_log.jsonl by the first run
PUBLISHED_MANIFEST_SHA256 = "687216575424d7e39723161a8d7bc5932d06112459363b6eaf81bf4e50fc9cd4"
PUBLISHED = ["hotels", "wildlife", "f1-dnf", "nba", "elec", "nebraska-weather", "nycflights", "bank-full"]


def screens_all_cuts(d):
    """The first plan's single-feature screen (label_auc, drift_auc), on each of the three cuts. Cut 1 is the
    first plan's screen; cuts 2 and 3 are an addition that catches columns whose coding changes later."""
    g = base.geometry(d.t.values)
    X = base.design(d, base.features(d), base.categoricals(d))
    return pd.concat([base.screen(X, d.y.values, dict(g, e2=[g["e2"][k]], test=[g["test"][k]])).assign(cut=k + 1)
                      for k in range(3)], ignore_index=True)


def assert_published(folder):
    """The first run's manifest and the thin files of its eight published datasets are unchanged."""
    assert base.sha256(f"{folder}/_manifest.json") == PUBLISHED_MANIFEST_SHA256, "published manifest changed"
    recorded = json.load(open(f"{folder}/_manifest.json"))["prepared"]
    for name in PUBLISHED:
        assert base.sha256(f"{folder}/{name}_thin.csv") == recorded[f"{name}_thin.csv"], f"{name} thin file changed"
    return {f"{name}_thin.csv": recorded[f"{name}_thin.csv"] for name in PUBLISHED}


def main():
    data, out_dir, published_dir = sys.argv[1:4]
    only = sys.argv[4:]
    published = assert_published(published_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, inputs = prep(data)
    rows, prepared = [], {}
    for name, (df, pos_name, neg_name) in frames.items():
        if only and name not in only:
            continue
        if df.y.mean() > 0.5:
            df = df.assign(y=1 - df.y)
            pos_name = neg_name
        share_full = df.y.mean()
        d = base.populations(df)["thin"]
        row, scr = base.run_checks(name, "thin", d, pos_name, share_full, len(df))
        rows.append(row)
        if row["passes"]:
            d.to_csv(out_dir / f"{name}_thin.csv", index=False)
            prepared[f"{name}_thin.csv"] = base.sha256(out_dir / f"{name}_thin.csv")
            json.dump({"categorical": base.categoricals(d), "features": base.features(d), "class": pos_name,
                       "share": round(float(share_full), 4), "sha256": {f"{name}_thin.csv": prepared[f"{name}_thin.csv"]}},
                      open(out_dir / f"{name}.json", "w"), indent=1)
        scr.assign(dataset=name).to_csv(out_dir / f"_features_{name}.csv", index=False)
        allcuts = screens_all_cuts(d)
        allcuts.assign(dataset=name).to_csv(out_dir / f"_features_allcuts_{name}.csv", index=False)
        extra = allcuts[(allcuts.cut > 1) & ((allcuts.label_auc > 0.8) | (allcuts.drift_auc > 0.8))]
        if len(extra):
            print(f"all-cuts flags {name}: " + "; ".join(
                f"{r.feature} cut {r.cut} label {r.label_auc:.2f} drift {r.drift_auc:.2f}" for r in extra.itertuples()))
    s = pd.DataFrame(rows)
    path = out_dir / "_summary_thin_ext.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.dataset.isin(s.dataset)]
        s.to_csv(path, index=False) if old.empty else pd.concat([old, s]).to_csv(path, index=False)
    else:
        s.to_csv(path, index=False)
    pd.set_option("display.width", 300, "display.max_colwidth", 60, "display.max_columns", 50)
    triple = lambda p: s[[f"{p}_c{k}" for k in (1, 2, 3)]].round(3).astype(str).agg("/".join, axis=1)
    print(pd.DataFrame({
        "dataset": s.dataset, "rows": s.rows, "span": s.span, "class (full/pop %)": s["class"] + " " + s.share_full.astype(str) + "/" + s.share_pop.astype(str),
        "test min": s.test.map(lambda x: "/".join(map(str, x))), "val": s.val_min,
        "fit": np.minimum(s.fit_e2, s.fit_e1), "ef": np.minimum(s.ef_e2, s.ef_e1), "er": np.minimum(s.er_e2, s.er_e1),
        "auc time": triple("auc_time"), "auc rand": triple("auc_rand"), "low": s.low_signal, "persist": s.persistence,
        "pass": s.passes, "reason": s.reason, "flags": s["flags"], "check 5": s.proxies}).to_string(index=False))
    import datetime
    import lightgbm, scipy, sklearn
    downloaded = {rel: datetime.date.fromtimestamp((Path(data) / rel).stat().st_birthtime).isoformat() for rel in inputs}
    modified = {rel: datetime.date.fromtimestamp((Path(data) / rel).stat().st_mtime).isoformat() for rel in inputs}
    samples = {p.name: base.sha256(p) for p in sorted(Path(data).glob("*_sample.csv"))}
    packages = {m.__name__: m.__version__ for m in (np, pd, lightgbm, scipy, sklearn)}
    json.dump({"inputs": inputs, "downloaded": downloaded, "file_modified": modified, "samples": samples, "packages": packages,
               "prepared": prepared, "published_manifest_sha256": PUBLISHED_MANIFEST_SHA256,
               "published": published}, open(out_dir / "_manifest_ext.json", "w"), indent=1)


if __name__ == "__main__":
    main()
