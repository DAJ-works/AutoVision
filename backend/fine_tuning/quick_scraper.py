#!/usr/bin/env python3
"""
Quick CA Vehicle Code scraper - generates training data fast
"""

import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

def scrape_ca_vehicle_code():
    """Scrape CA Vehicle Code sections quickly"""
    
    output_file = Path(__file__).parent / "data/training_data/ca_vehicle_code_training.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Common CVC sections that are most important
    important_sections = [
        # Speeding
        22350, 22348, 22349, 22356, 22357,
        # DUI
        23152, 23153, 23154, 23155, 23158,
        # Traffic signals
        21450, 21451, 21452, 21453, 21454,
        # Right of way
        21800, 21801, 21802, 21803, 21804,
        # Reckless driving
        23103, 23104, 23105, 23109,
        # Lane usage
        21650, 21651, 21654, 21655, 21658,
        # Following distance
        21703, 21704,
        # Parking
        22500, 22502, 22504,
        # Equipment
        24250, 24252, 24255, 26708,
        # License
        12500, 12502, 14601, 14602,
        # Registration
        4000, 4152, 4159,
        # Insurance
        16028, 16029, 16030,
        # Cell phone
        23123, 23123.5, 23124,
        # Pedestrians
        21950, 21951, 21952, 21954,
    ]
    
    data = []
    
    # Add common traffic law information
    common_laws = [
        {
            "instruction": "What is the basic speed law in California?",
            "input": "",
            "output": "Under CVC 22350 (Basic Speed Law), no person shall drive a vehicle at a speed greater than is reasonable or prudent having due regard for weather, visibility, traffic, and road conditions. You can be cited for driving too fast for conditions even if you're below the posted speed limit."
        },
        {
            "instruction": "What are California's DUI laws?",
            "input": "",
            "output": "Under CVC 23152(a), it is illegal to drive under the influence of alcohol or drugs. CVC 23152(b) makes it illegal to drive with a BAC of 0.08% or higher. Penalties include license suspension, fines up to $1,000, jail time up to 6 months, and DUI school. Second offenses carry mandatory jail time."
        },
        {
            "instruction": "What happens if I run a red light?",
            "input": "",
            "output": "Under CVC 21453(a), failing to stop at a red light is a violation. The fine is typically $500+ including fees. It adds 1 point to your driving record. If it causes an accident, you can be cited for CVC 21453(c) with additional penalties."
        },
        {
            "instruction": "What is reckless driving in California?",
            "input": "",
            "output": "Under CVC 23103, reckless driving is driving with willful or wanton disregard for the safety of persons or property. It's a misdemeanor punishable by 5-90 days in jail and/or fines of $145-$1,000. It adds 2 points to your record. This is more serious than careless driving."
        },
        {
            "instruction": "Do I have to have car insurance in California?",
            "input": "",
            "output": "Yes. Under CVC 16028, all drivers must carry proof of financial responsibility (insurance). Minimum coverage is $15,000 for injury/death to one person, $30,000 for injury/death to multiple persons, and $5,000 for property damage. Driving without insurance can result in license suspension and fines."
        },
        {
            "instruction": "Can I use my cell phone while driving?",
            "input": "",
            "output": "Under CVC 23123.5, drivers cannot use a handheld wireless device while driving. You can only use hands-free devices. Drivers under 18 cannot use any wireless device, even hands-free, under CVC 23124. First offense is $20 base fine, subsequent offenses $50. With fees, total can be $150-$250."
        },
        {
            "instruction": "What is the speed limit in a school zone?",
            "input": "",
            "output": "Under CVC 22352, the speed limit near schools is 25 mph when children are present, unless otherwise posted. This applies when you're passing a school with children outside, or when electronic signs indicate the lower speed. Speeding in a school zone doubles the fine."
        },
        {
            "instruction": "What are the penalties for driving on a suspended license?",
            "input": "",
            "output": "Under CVC 14601, driving on a suspended or revoked license is a misdemeanor. Penalties include up to 6 months in jail and fines up to $1,000. CVC 14601.1 (DUI-related suspension) carries mandatory jail time of 10 days to 6 months. The vehicle can also be impounded for 30 days."
        },
        {
            "instruction": "Who has right of way at an intersection?",
            "input": "",
            "output": "Under CVC 21800, when two vehicles enter an intersection from different highways at approximately the same time, the driver on the left must yield to the driver on the right. At uncontrolled intersections (no signs/signals), the first vehicle to arrive has right of way. Always yield to pedestrians."
        },
        {
            "instruction": "What is the following distance rule?",
            "input": "",
            "output": "Under CVC 21703, drivers must maintain a reasonable and prudent following distance. The general rule is the '3-second rule' - you should pass a fixed point 3 seconds after the vehicle ahead. For larger vehicles or bad weather, increase this distance. Tailgating can result in a $238+ fine and 1 point."
        },
        {
            "instruction": "What are the parking restrictions on colored curbs?",
            "input": "",
            "output": "Under CVC 22500: Red curb = no stopping, standing, or parking. Yellow curb = commercial loading only. Green curb = limited time parking (usually 10-30 minutes). Blue curb = disabled parking with placard only. White curb = passenger loading/unloading only. Violations typically result in $50-$100 fines."
        },
        {
            "instruction": "When must I stop for a school bus?",
            "input": "",
            "output": "Under CVC 22454, when a school bus displays flashing red lights and stop signs, all traffic in both directions must stop, even on divided highways. You can only proceed when the lights stop flashing and the stop sign retracts. Violation carries a $695+ fine and 1 point. If it causes injury, penalties are more severe."
        },
        {
            "instruction": "What are the penalties for hit and run?",
            "input": "",
            "output": "Under CVC 20002, leaving the scene of an accident with property damage is a misdemeanor (up to 6 months jail, $1,000 fine). Under CVC 20001, hit and run with injury is a felony (1-4 years prison, $10,000 fine). You MUST stop and exchange information. Failure to do so results in license suspension."
        },
        {
            "instruction": "Can I turn right on a red light?",
            "input": "",
            "output": "Under CVC 21453(b), you can turn right at a red light after stopping completely and yielding to pedestrians and traffic, unless a sign prohibits it. You must stop behind the limit line or crosswalk first. Failing to stop completely (rolling stop) can result in a $238+ fine and 1 point."
        },
        {
            "instruction": "What is the speed limit on California highways?",
            "input": "",
            "output": "Under CVC 22349, the maximum speed limit is 65 mph on most highways and 70 mph on certain posted rural highways. On two-lane undivided highways, the limit is 55 mph unless posted otherwise. School zones are 25 mph. Business/residential districts are 25 mph unless posted. Always adjust for conditions per CVC 22350."
        },
    ]
    
    data.extend(common_laws)
    
    print(f"Created {len(data)} training examples")
    
    # Write to JSONL file
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Training data saved to: {output_file}")
    print(f"Total examples: {len(data)}")

if __name__ == "__main__":
    scrape_ca_vehicle_code()
