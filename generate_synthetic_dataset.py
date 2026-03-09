#!/usr/bin/env python3
"""
Synthetic Voice Command Dataset Generator

Generates audio + tool-call pairs for training a speech-to-tool-call model.
Uses Edge TTS for speech synthesis.

30 functions matching the STOP dataset domains:
  Alarm (6), Weather (1), Reminder (4), Timer (6),
  Messaging (3), Music (5), Navigation (5), Event (1)
"""

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

VOICES = {
    "train": [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-ChristopherNeural",
        "en-US-EricNeural",
        "en-US-MichelleNeural",
        "en-US-RogerNeural",
        "en-US-SteffanNeural",
    ],
    "val": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "test": ["en-AU-NatashaNeural", "en-AU-WilliamMultilingualNeural"],
}

# -----------------------------------------------------------------------------
# SHARED ENTITY POOLS
# -----------------------------------------------------------------------------

TIMES = [
    "7 AM", "7:30 AM", "8 AM", "8:15 AM", "8:30 AM", "9 AM", "9:30 AM",
    "10 AM", "10:30 AM", "11 AM", "11:30 AM", "noon", "12:30 PM",
    "1 PM", "1:30 PM", "2 PM", "2:30 PM", "3 PM", "3:30 PM", "4 PM",
    "4:30 PM", "5 PM", "5:30 PM", "6 PM", "6:30 PM", "7 PM", "7:30 PM",
    "8 PM", "8:30 PM", "9 PM", "9:30 PM", "10 PM", "10:30 PM", "11 PM",
    "midnight", "5 AM", "5:30 AM", "6 AM", "6:30 AM",
]

DATES = [
    "tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "next Monday", "next Friday", "this weekend",
    "the day after tomorrow", "January 5th", "February 14th", "March 1st",
    "April 10th", "May 20th", "June 15th", "July 4th", "August 8th",
    "September 1st", "October 31st", "November 25th", "December 25th",
    "next week", "in two days", "tonight", "this afternoon", "this evening",
    "this morning", "later today",
]

ALARM_NAMES = [
    "morning alarm", "work alarm", "wake up alarm", "school alarm",
    "nap alarm", "workout alarm", "medication alarm", "meeting alarm",
    "first alarm", "second alarm", "backup alarm", "early alarm",
    "late alarm", "weekend alarm", "weekday alarm",
]

DURATIONS = [
    "5 minutes", "10 minutes", "15 minutes", "20 minutes", "30 minutes",
    "45 minutes", "1 hour", "1 and a half hours", "2 hours", "3 hours",
    "90 seconds", "2 minutes", "3 minutes", "25 minutes", "40 minutes",
    "1 minute", "4 hours", "5 hours", "half an hour",
]

TIMER_NAMES = [
    "cooking timer", "egg timer", "workout timer", "study timer",
    "laundry timer", "nap timer", "pasta timer", "pizza timer",
    "meditation timer", "break timer", "oven timer", "tea timer",
    "pomodoro timer", "game timer", "chicken timer",
]

REMINDER_TEXTS = [
    "take out the trash", "pick up groceries", "call the dentist",
    "pay the electric bill", "water the plants", "feed the dog",
    "pick up the kids", "take my medication", "go to the gym",
    "submit the report", "buy birthday present", "schedule oil change",
    "return library books", "clean the house", "do laundry",
    "send the invoice", "book a flight", "renew my license",
    "call Mom", "check the mail", "walk the dog", "mow the lawn",
    "wash the car", "buy flowers", "drop off dry cleaning",
    "cancel the subscription", "fix the leaky faucet", "change air filter",
    "order new contacts", "schedule haircut", "prep dinner",
    "file taxes", "update my resume", "buy cat food", "restock vitamins",
    "charge my laptop", "set up the printer", "organize the garage",
    "replace batteries", "defrost the chicken",
]

CONTACTS = [
    "John", "Sarah", "Mike", "Emily", "David", "Jessica", "James", "Jennifer",
    "Robert", "Lisa", "Michael", "Amanda", "William", "Ashley", "Daniel",
    "Stephanie", "Christopher", "Nicole", "Matthew", "Megan", "Andrew",
    "Rachel", "Joshua", "Lauren", "Ryan", "Hannah", "Brandon", "Samantha",
    "Carlos", "Maria", "Raj", "Priya", "Wei", "Yuki", "Alex", "Olivia",
    "Mom", "Dad", "my wife", "my husband", "my brother", "my sister",
    "grandma", "grandpa", "my boss", "the team",
]

MESSAGES = [
    "I'll be there in 10 minutes", "Running late", "On my way",
    "Can you pick up milk", "Meeting at 3", "Happy birthday",
    "Thanks for dinner", "See you tomorrow", "Call me when you can",
    "I'm leaving now", "Don't forget the keys", "Good morning",
    "I love you", "Sounds good", "Let me know when you're ready",
    "I'll be home by 6", "Can we reschedule", "Got it thanks",
    "Where are you", "Almost there", "I miss you",
    "Dinner is ready", "I'm at the store", "What time works for you",
    "Sorry I missed your call", "Can you send me the address",
]

REACTIONS = [
    "thumbs up", "heart", "laugh", "like", "love", "haha",
    "wow", "sad", "angry", "celebrate", "fire", "clap",
]

SONGS = [
    "Bohemian Rhapsody", "Shape of You", "Blinding Lights", "Despacito",
    "Old Town Road", "Uptown Funk", "Rolling in the Deep", "Lose Yourself",
    "Billie Jean", "Hotel California", "Imagine", "Wonderwall",
    "Smells Like Teen Spirit", "Hey Jude", "Stairway to Heaven",
    "Sweet Child O'Mine", "Thriller", "Like a Prayer", "Purple Rain",
    "Mr. Brightside", "Somebody That I Used to Know", "Happy", "Get Lucky",
    "All of Me", "Shallow", "Bad Guy", "Levitating", "As It Was",
    "Anti-Hero", "Flowers", "Cruel Summer", "Dance Monkey", "Havana",
    "Drivers License", "Stay", "Good 4 U", "Watermelon Sugar",
    "Counting Stars", "Thinking Out Loud", "Take Me to Church",
]

ARTISTS = [
    "Taylor Swift", "The Beatles", "Drake", "Ed Sheeran", "Beyonce", "Adele",
    "Billie Eilish", "Bad Bunny", "The Weeknd", "Harry Styles", "Dua Lipa",
    "Post Malone", "Ariana Grande", "Bruno Mars", "Doja Cat", "Olivia Rodrigo",
    "SZA", "Kendrick Lamar", "Rihanna", "Justin Bieber", "Lady Gaga",
    "Coldplay", "BTS", "Miley Cyrus", "Sam Smith", "Elton John", "Eminem",
    "Queen", "Fleetwood Mac", "Michael Jackson", "Prince", "Radiohead",
    "Arctic Monkeys", "Imagine Dragons", "Twenty One Pilots", "Green Day",
    "Led Zeppelin", "Pink Floyd", "Nirvana", "Red Hot Chili Peppers",
]

GENRES = [
    "rock", "pop", "jazz", "classical", "hip hop", "country", "electronic",
    "R&B", "reggae", "folk", "metal", "blues", "soul", "Latin",
    "indie", "alternative", "lo-fi", "acoustic", "punk", "dance",
]

PLAYLISTS = [
    "favorites", "workout", "chill", "party", "sleep", "road trip",
    "morning", "focus", "running", "dinner", "throwbacks", "top hits",
    "discover weekly", "release radar", "jazz vibes", "acoustic",
    "study music", "relaxing", "summer vibes", "cooking playlist",
]

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "San Francisco", "Seattle", "Denver", "Washington DC", "Nashville",
    "Boston", "Portland", "Las Vegas", "Miami", "Atlanta", "Minneapolis",
    "Detroit", "Salt Lake City", "London", "Paris", "Tokyo", "Sydney",
    "Toronto", "Berlin", "Madrid", "Rome", "Amsterdam", "Dubai", "Singapore",
    "Mumbai", "Bangkok", "Seoul", "Mexico City", "Dublin", "Vienna",
    "Prague", "Stockholm", "Barcelona", "Lisbon", "Copenhagen",
    "here", "my area", "around here", "downtown", "outside",
]

DESTINATIONS = [
    "home", "work", "the airport", "nearest gas station", "nearest hospital",
    "downtown", "the mall", "grocery store", "the gym", "the pharmacy",
    "the library", "the park", "school", "the office", "the train station",
    "the bus stop", "the beach", "the museum", "church", "the post office",
    "the bank", "the dentist", "the doctor's office", "the university",
    "the stadium", "the movie theater", "Target", "Walmart", "Costco",
    "Home Depot", "the coffee shop", "my parents' house", "the hotel",
    "the restaurant", "the convention center", "daycare", "the vet",
    "the car wash", "the barber shop", "the salon",
]

TRAVEL_MODES = [
    "driving", "walking", "biking", "transit", "by car", "on foot",
    "by bus", "by train", "by bike",
]

EVENT_NAMES = [
    "concerts", "festivals", "movies", "shows", "games", "events",
    "exhibits", "plays", "performances", "meetups", "workshops",
    "conferences", "parties", "fundraisers", "farmers markets",
]

EVENT_CATEGORIES = [
    "music", "sports", "art", "food", "tech", "comedy", "theater",
    "dance", "film", "photography", "fashion", "charity", "outdoor",
    "family", "holiday", "cultural", "educational",
]

WEATHER_ATTRIBUTES = [
    "temperature", "humidity", "wind", "rain", "snow", "forecast",
    "UV index", "air quality", "pollen count", "visibility",
]

# -----------------------------------------------------------------------------
# TOOL DEFINITIONS - 30 FUNCTIONS (matching STOP domains)
# -----------------------------------------------------------------------------

TOOLS = {}

# =============================================================================
# DOMAIN: ALARM (6 functions)
# =============================================================================

TOOLS["alarm_set"] = {
    "domain": "alarm",
    "description": "Set a new alarm",
    "parameters": {
        "time": {"type": "string", "description": "Time for the alarm"},
        "name": {"type": "string", "description": "Alarm label"},
        "recurrence": {"type": "string", "description": "Recurring schedule"},
    },
    "required": ["time"],
    "phrases": [
        "Set an alarm for {time}",
        "Wake me up at {time}",
        "Set an alarm for {time} {date}",
        "Create an alarm for {time}",
        "Alarm at {time}",
        "I need an alarm for {time}",
        "Set a {name} for {time}",
        "Set an alarm for {time} every {recurrence}",
        "Can you set an alarm for {time}",
        "Set my alarm for {time} tomorrow",
        "I want to wake up at {time}",
        "Please set an alarm at {time}",
    ],
    "param_values": {
        "time": TIMES,
        "date": DATES,
        "name": ALARM_NAMES,
        "recurrence": ["weekday", "weekend", "day", "Monday", "Monday through Friday",
                       "week", "morning"],
    },
}

TOOLS["alarm_check"] = {
    "domain": "alarm",
    "description": "Check existing alarms",
    "parameters": {
        "time": {"type": "string"},
        "name": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "What alarms do I have",
        "Show me my alarms",
        "When is my alarm set for",
        "Do I have any alarms",
        "What time is my alarm",
        "Check my alarms",
        "When does my alarm go off",
        "List my alarms",
        "Are there any alarms set",
        "What's my next alarm",
    ],
    "param_values": {
        "time": TIMES,
        "name": ALARM_NAMES,
    },
}

TOOLS["alarm_cancel"] = {
    "domain": "alarm",
    "description": "Delete an alarm",
    "parameters": {
        "time": {"type": "string"},
        "name": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Delete my {time} alarm",
        "Cancel my alarm",
        "Remove the {time} alarm",
        "Turn off my alarm",
        "Delete all alarms",
        "Cancel the {name}",
        "Remove my alarms",
        "Get rid of my {time} alarm",
        "I don't need my {time} alarm anymore",
        "Delete the alarm for {time}",
    ],
    "param_values": {
        "time": TIMES,
        "name": ALARM_NAMES,
    },
}

TOOLS["alarm_update"] = {
    "domain": "alarm",
    "description": "Change an existing alarm",
    "parameters": {
        "time": {"type": "string", "description": "Current alarm time"},
        "new_time": {"type": "string", "description": "New alarm time"},
        "name": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Change my alarm to {new_time}",
        "Move my {time} alarm to {new_time}",
        "Update my alarm to {new_time}",
        "Reschedule my alarm to {new_time}",
        "Change the {name} to {new_time}",
        "Push my alarm back to {new_time}",
        "Set my alarm earlier to {new_time}",
        "Move my alarm from {time} to {new_time}",
    ],
    "param_values": {
        "time": TIMES,
        "new_time": TIMES,
        "name": ALARM_NAMES,
    },
}

TOOLS["alarm_snooze"] = {
    "domain": "alarm",
    "description": "Snooze a ringing alarm",
    "parameters": {
        "duration": {"type": "string", "description": "Snooze duration"},
    },
    "required": [],
    "phrases": [
        "Snooze",
        "Snooze for {duration}",
        "5 more minutes",
        "10 more minutes",
        "Snooze my alarm",
        "Snooze it",
        "Give me {duration} more",
        "Let me sleep {duration} more",
        "Hit snooze",
        "Snooze the alarm for {duration}",
        "Just a few more minutes",
        "Snooze for another {duration}",
    ],
    "param_values": {
        "duration": ["5 minutes", "10 minutes", "15 minutes", "20 minutes",
                     "30 minutes", "a few minutes", "half an hour"],
    },
}

TOOLS["alarm_silence"] = {
    "domain": "alarm",
    "description": "Silence a ringing alarm",
    "parameters": {
        "name": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Stop the alarm",
        "Turn off the alarm",
        "Silence the alarm",
        "Shut off the alarm",
        "Dismiss the alarm",
        "Turn it off",
        "Stop ringing",
        "Quiet",
        "Shut up",
        "Make it stop",
        "Kill the alarm",
        "Dismiss it",
    ],
    "param_values": {
        "name": ALARM_NAMES,
    },
}

# =============================================================================
# DOMAIN: WEATHER (1 function)
# =============================================================================

TOOLS["weather_query"] = {
    "domain": "weather",
    "description": "Get weather information",
    "parameters": {
        "location": {"type": "string"},
        "time": {"type": "string"},
        "attribute": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "What's the weather in {location}",
        "How's the weather in {location}",
        "What's the {attribute} in {location}",
        "Is it going to rain {time}",
        "Weather forecast for {time}",
        "What's the weather like {time}",
        "Will it snow {time}",
        "What's the temperature in {location}",
        "How hot is it in {location}",
        "How cold is it in {location}",
        "What's the forecast for {location} {time}",
        "Do I need an umbrella {time}",
        "When is {attribute} in {location}",
        "What time is {attribute} {time}",
        "What's the weather",
        "Is it going to be nice {time}",
        "What should I wear {time}",
        "How's the weather outside",
        "Check the weather for {location}",
    ],
    "param_values": {
        "location": LOCATIONS,
        "time": DATES,
        "attribute": WEATHER_ATTRIBUTES + ["sunset", "sunrise"],
    },
}

# =============================================================================
# DOMAIN: REMINDER (4 functions)
# =============================================================================

TOOLS["reminder_set"] = {
    "domain": "reminder",
    "description": "Create a new reminder",
    "parameters": {
        "text": {"type": "string", "description": "What to be reminded about"},
        "time": {"type": "string"},
        "person": {"type": "string"},
        "location": {"type": "string"},
    },
    "required": ["text"],
    "phrases": [
        "Remind me to {text}",
        "Remind me to {text} at {time}",
        "Remind me to {text} {date}",
        "Set a reminder to {text}",
        "Create a reminder to {text} at {time}",
        "Remind me to {text} when I get to {location}",
        "Don't let me forget to {text}",
        "Remind {person} to {text}",
        "Remind me to {text} at {time} {date}",
        "I need a reminder to {text}",
        "Can you remind me to {text}",
        "Remember to {text} at {time}",
    ],
    "param_values": {
        "text": REMINDER_TEXTS,
        "time": TIMES,
        "date": DATES,
        "person": CONTACTS,
        "location": ["home", "work", "the store", "the gym", "school",
                     "the office", "the pharmacy", "the airport"],
    },
}

TOOLS["reminder_check"] = {
    "domain": "reminder",
    "description": "Check existing reminders",
    "parameters": {
        "text": {"type": "string"},
        "person": {"type": "string"},
        "time": {"type": "string"},
        "query": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "What reminders do I have",
        "Show me my reminders",
        "Do I have any reminders",
        "What am I supposed to do {date}",
        "Check my reminders",
        "What's on my reminder list",
        "When is my reminder to {text}",
        "How many reminders do I have",
        "List my reminders for {date}",
        "Any reminders for {date}",
        "What reminders are set for {date}",
    ],
    "param_values": {
        "text": REMINDER_TEXTS,
        "person": CONTACTS,
        "time": TIMES,
        "date": DATES,
        "query": ["date_time", "count", "location"],
    },
}

TOOLS["reminder_cancel"] = {
    "domain": "reminder",
    "description": "Delete a reminder",
    "parameters": {
        "text": {"type": "string"},
        "time": {"type": "string"},
        "person": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Delete my reminder to {text}",
        "Cancel my reminder to {text}",
        "Remove the reminder to {text}",
        "Delete all reminders",
        "I don't need the reminder to {text}",
        "Cancel the reminder for {date}",
        "Remove my reminders",
        "Get rid of the reminder to {text}",
        "Delete the {date} reminder",
    ],
    "param_values": {
        "text": REMINDER_TEXTS,
        "time": TIMES,
        "date": DATES,
        "person": CONTACTS,
    },
}

TOOLS["reminder_update"] = {
    "domain": "reminder",
    "description": "Update an existing reminder",
    "parameters": {
        "text": {"type": "string"},
        "time": {"type": "string"},
        "new_time": {"type": "string"},
        "new_text": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Change my reminder to {text} to {new_time}",
        "Move my reminder to {new_time}",
        "Update the reminder to {text}",
        "Reschedule my reminder to {new_time}",
        "Change the reminder from {text} to {new_text}",
        "Push my reminder to {new_time}",
        "Update my {date} reminder to {new_time}",
    ],
    "param_values": {
        "text": REMINDER_TEXTS,
        "time": TIMES,
        "date": DATES,
        "new_time": TIMES,
        "new_text": REMINDER_TEXTS,
    },
}

# =============================================================================
# DOMAIN: TIMER (6 functions)
# =============================================================================

TOOLS["timer_set"] = {
    "domain": "timer",
    "description": "Set a new timer",
    "parameters": {
        "duration": {"type": "string"},
        "name": {"type": "string"},
    },
    "required": ["duration"],
    "phrases": [
        "Set a timer for {duration}",
        "Start a {duration} timer",
        "Timer for {duration}",
        "Set a {name} for {duration}",
        "Count down {duration}",
        "{duration} timer",
        "Start a timer for {duration}",
        "Can you set a timer for {duration}",
        "I need a timer for {duration}",
        "Set a {duration} countdown",
    ],
    "param_values": {
        "duration": DURATIONS,
        "name": TIMER_NAMES,
    },
}

TOOLS["timer_check"] = {
    "domain": "timer",
    "description": "Check active timers",
    "parameters": {
        "name": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "How much time is left",
        "Check my timer",
        "How long is left on the timer",
        "What's the time on my timer",
        "Timer status",
        "How much time is left on the {name}",
        "Is the timer still running",
        "Show my timers",
        "How many minutes left",
        "When does my timer go off",
    ],
    "param_values": {
        "name": TIMER_NAMES,
    },
}

TOOLS["timer_cancel"] = {
    "domain": "timer",
    "description": "Cancel a timer",
    "parameters": {
        "name": {"type": "string"},
        "duration": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Cancel my timer",
        "Delete the timer",
        "Stop the timer",
        "Cancel the {name}",
        "Remove the timer",
        "Cancel all timers",
        "Get rid of the timer",
        "Delete the {name}",
        "I don't need the timer anymore",
    ],
    "param_values": {
        "name": TIMER_NAMES,
        "duration": DURATIONS,
    },
}

TOOLS["timer_pause"] = {
    "domain": "timer",
    "description": "Pause a running timer",
    "parameters": {
        "name": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Pause the timer",
        "Pause my timer",
        "Hold the timer",
        "Pause the {name}",
        "Freeze the timer",
        "Stop counting",
        "Pause it",
        "Hold on the timer",
    ],
    "param_values": {
        "name": TIMER_NAMES,
    },
}

TOOLS["timer_resume"] = {
    "domain": "timer",
    "description": "Resume or restart a timer",
    "parameters": {
        "name": {"type": "string"},
        "action": {"type": "string", "enum": ["resume", "restart"]},
    },
    "required": [],
    "phrases": [
        ("Resume the timer", {"action": "resume"}),
        ("Continue the timer", {"action": "resume"}),
        ("Unpause the timer", {"action": "resume"}),
        ("Resume my timer", {"action": "resume"}),
        ("Start the timer again", {"action": "resume"}),
        ("Restart the timer", {"action": "restart"}),
        ("Reset the timer", {"action": "restart"}),
        ("Restart the {name}", {"action": "restart"}),
        ("Start the timer over", {"action": "restart"}),
        ("Resume the {name}", {"action": "resume"}),
    ],
    "param_values": {
        "name": TIMER_NAMES,
        "action": ["resume", "restart"],
    },
}

TOOLS["timer_modify"] = {
    "domain": "timer",
    "description": "Add or subtract time from a timer",
    "parameters": {
        "duration": {"type": "string"},
        "name": {"type": "string"},
        "action": {"type": "string", "enum": ["add", "subtract", "set"]},
    },
    "required": ["duration"],
    "phrases": [
        ("Add {duration} to the timer", {"action": "add"}),
        ("Add {duration} more", {"action": "add"}),
        ("Put {duration} more on the timer", {"action": "add"}),
        ("Give me {duration} more on the timer", {"action": "add"}),
        ("Take {duration} off the timer", {"action": "subtract"}),
        ("Subtract {duration} from the timer", {"action": "subtract"}),
        ("Remove {duration} from the timer", {"action": "subtract"}),
        ("Change the timer to {duration}", {"action": "set"}),
        ("Update the timer to {duration}", {"action": "set"}),
        ("Set the timer to {duration}", {"action": "set"}),
        ("Add {duration} to the {name}", {"action": "add"}),
    ],
    "param_values": {
        "duration": DURATIONS,
        "name": TIMER_NAMES,
        "action": ["add", "subtract", "set"],
    },
}

# =============================================================================
# DOMAIN: MESSAGING (3 functions)
# =============================================================================

TOOLS["message_send"] = {
    "domain": "messaging",
    "description": "Send a message",
    "parameters": {
        "recipient": {"type": "string"},
        "text": {"type": "string"},
        "group": {"type": "string"},
    },
    "required": ["recipient"],
    "phrases": [
        "Send a message to {recipient}",
        "Text {recipient} {text}",
        "Tell {recipient} {text}",
        "Message {recipient} saying {text}",
        "Send {recipient} a text",
        "Let {recipient} know {text}",
        "Send a message to {recipient} saying {text}",
        "Text {recipient}",
        "Write a message to {recipient}",
        "Reply to {recipient} with {text}",
        "Shoot {recipient} a text saying {text}",
    ],
    "param_values": {
        "recipient": CONTACTS,
        "text": MESSAGES,
        "group": ["family group", "work chat", "friends", "team", "roommates"],
    },
}

TOOLS["message_read"] = {
    "domain": "messaging",
    "description": "Read messages",
    "parameters": {
        "sender": {"type": "string"},
        "time": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Read my messages",
        "Any new messages",
        "Check my texts",
        "Read messages from {sender}",
        "What did {sender} say",
        "Do I have any messages",
        "Show me my texts",
        "What messages do I have from {sender}",
        "Read my latest messages",
        "Any texts from {sender}",
    ],
    "param_values": {
        "sender": CONTACTS,
        "time": DATES,
    },
}

TOOLS["message_react"] = {
    "domain": "messaging",
    "description": "React to a message",
    "parameters": {
        "reaction": {"type": "string"},
        "sender": {"type": "string"},
    },
    "required": [],
    "phrases": [
        ("React with {reaction}", {}),
        ("{reaction} that message", {}),
        ("Send a {reaction} to {sender}'s message", {}),
        ("Like {sender}'s message", {"reaction": "like"}),
        ("Heart that", {"reaction": "heart"}),
        ("Laugh at {sender}'s message", {"reaction": "haha"}),
        ("Thumbs up", {"reaction": "thumbs up"}),
        ("React to {sender}'s message with {reaction}", {}),
        ("Give that a {reaction}", {}),
    ],
    "param_values": {
        "reaction": REACTIONS,
        "sender": CONTACTS,
    },
}

# =============================================================================
# DOMAIN: MUSIC (5 functions)
# =============================================================================

TOOLS["music_play"] = {
    "domain": "music",
    "description": "Play music",
    "parameters": {
        "song": {"type": "string"},
        "artist": {"type": "string"},
        "genre": {"type": "string"},
        "playlist": {"type": "string"},
        "album": {"type": "string"},
        "action": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "Play {song}",
        "Play {song} by {artist}",
        "Play some {genre} music",
        "Play songs by {artist}",
        "Play my {playlist} playlist",
        "Put on some {genre}",
        "I want to hear {artist}",
        "Play music",
        "Play some music",
        ("Play that song again", {"action": "replay"}),
        ("Replay this song", {"action": "replay"}),
        ("Play that again", {"action": "replay"}),
        ("Start this song over", {"action": "replay"}),
    ],
    "param_values": {
        "song": SONGS,
        "artist": ARTISTS,
        "genre": GENRES,
        "playlist": PLAYLISTS,
        "album": ["Abbey Road", "Thriller", "1989", "Rumours", "Back in Black",
                  "The Dark Side of the Moon", "Nevermind", "OK Computer"],
        "action": ["replay"],
    },
}

TOOLS["music_skip"] = {
    "domain": "music",
    "description": "Skip or go to previous track",
    "parameters": {
        "direction": {"type": "string", "enum": ["next", "previous"]},
        "count": {"type": "integer"},
    },
    "required": [],
    "phrases": [
        ("Next song", {"direction": "next"}),
        ("Skip this song", {"direction": "next"}),
        ("Skip this", {"direction": "next"}),
        ("Next track", {"direction": "next"}),
        ("Go to the next song", {"direction": "next"}),
        ("Skip it", {"direction": "next"}),
        ("Skip ahead", {"direction": "next"}),
        ("Previous song", {"direction": "previous"}),
        ("Go back one song", {"direction": "previous"}),
        ("Play the previous track", {"direction": "previous"}),
        ("Previous track", {"direction": "previous"}),
        ("Go back", {"direction": "previous"}),
        ("Play the last song again", {"direction": "previous"}),
    ],
    "param_values": {
        "direction": ["next", "previous"],
        "count": [1, 2, 3],
    },
}

TOOLS["music_playback"] = {
    "domain": "music",
    "description": "Control music playback",
    "parameters": {
        "action": {"type": "string", "enum": ["pause", "stop", "shuffle", "loop", "like", "dislike"]},
        "song": {"type": "string"},
        "artist": {"type": "string"},
    },
    "required": ["action"],
    "phrases": [
        ("Pause the music", {"action": "pause"}),
        ("Pause", {"action": "pause"}),
        ("Pause it", {"action": "pause"}),
        ("Stop the music", {"action": "stop"}),
        ("Stop playing", {"action": "stop"}),
        ("Stop it", {"action": "stop"}),
        ("Shuffle my music", {"action": "shuffle"}),
        ("Turn on shuffle", {"action": "shuffle"}),
        ("Play on shuffle", {"action": "shuffle"}),
        ("Loop this song", {"action": "loop"}),
        ("Repeat this song", {"action": "loop"}),
        ("Put this on repeat", {"action": "loop"}),
        ("I like this song", {"action": "like"}),
        ("Thumbs up", {"action": "like"}),
        ("Like this", {"action": "like"}),
        ("I don't like this", {"action": "dislike"}),
        ("Thumbs down", {"action": "dislike"}),
        ("Dislike this song", {"action": "dislike"}),
    ],
    "param_values": {
        "action": ["pause", "stop", "shuffle", "loop", "like", "dislike"],
        "song": SONGS,
        "artist": ARTISTS,
    },
}

TOOLS["music_playlist"] = {
    "domain": "music",
    "description": "Manage playlists",
    "parameters": {
        "playlist": {"type": "string"},
        "song": {"type": "string"},
        "artist": {"type": "string"},
        "action": {"type": "string", "enum": ["add", "remove", "create"]},
    },
    "required": ["action"],
    "phrases": [
        ("Add this song to my {playlist} playlist", {"action": "add"}),
        ("Add {song} to {playlist}", {"action": "add"}),
        ("Put this in my {playlist} playlist", {"action": "add"}),
        ("Save this song to {playlist}", {"action": "add"}),
        ("Remove {song} from {playlist}", {"action": "remove"}),
        ("Take this song off {playlist}", {"action": "remove"}),
        ("Delete {song} from my {playlist} playlist", {"action": "remove"}),
        ("Create a playlist called {playlist}", {"action": "create"}),
        ("Make a new playlist named {playlist}", {"action": "create"}),
        ("Start a new {playlist} playlist", {"action": "create"}),
    ],
    "param_values": {
        "playlist": PLAYLISTS,
        "song": SONGS,
        "artist": ARTISTS,
        "action": ["add", "remove", "create"],
    },
}

# =============================================================================
# DOMAIN: NAVIGATION (5 functions)
# =============================================================================

TOOLS["nav_directions"] = {
    "domain": "navigation",
    "description": "Get or update directions",
    "parameters": {
        "destination": {"type": "string"},
        "origin": {"type": "string"},
        "mode": {"type": "string"},
        "action": {"type": "string"},
    },
    "required": ["destination"],
    "phrases": [
        "Get directions to {destination}",
        "Navigate to {destination}",
        "How do I get to {destination}",
        "Directions to {destination}",
        "Take me to {destination}",
        "Navigate to {destination} by {mode}",
        "Show me the way to {destination}",
        "Get directions from {origin} to {destination}",
        ("Find a different route to {destination}", {"action": "update"}),
        ("Avoid highways to {destination}", {"action": "update"}),
        ("Reroute to {destination}", {"action": "update"}),
    ],
    "param_values": {
        "destination": DESTINATIONS,
        "origin": ["home", "work", "here", "my current location", "the hotel",
                   "the airport", "downtown"],
        "mode": TRAVEL_MODES,
        "action": ["update"],
    },
}

TOOLS["nav_traffic"] = {
    "domain": "navigation",
    "description": "Check traffic conditions",
    "parameters": {
        "destination": {"type": "string"},
        "location": {"type": "string"},
        "route": {"type": "string"},
        "time": {"type": "string"},
        "query": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "How's traffic to {destination}",
        "Is there traffic on the way to {destination}",
        "Check traffic to {destination}",
        "Traffic report",
        "How's my commute looking",
        "Any traffic on {route}",
        "What's the traffic like right now",
        "Is there a lot of traffic",
        "How's the traffic to {destination} {time}",
        "Traffic on the way to work",
        ("How are the road conditions", {"query": "road_condition"}),
        ("What are the road conditions on {route}", {"query": "road_condition"}),
        ("Are the roads icy", {"query": "road_condition"}),
    ],
    "param_values": {
        "destination": DESTINATIONS,
        "location": LOCATIONS,
        "route": ["I-95", "the highway", "the freeway", "Route 1", "the turnpike",
                  "I-405", "I-10", "Route 66", "the expressway", "the bypass",
                  "I-75", "I-80", "the beltway"],
        "time": DATES,
        "query": ["road_condition"],
    },
}

TOOLS["nav_duration"] = {
    "domain": "navigation",
    "description": "Get estimated travel duration",
    "parameters": {
        "destination": {"type": "string"},
        "origin": {"type": "string"},
        "mode": {"type": "string"},
        "time": {"type": "string"},
    },
    "required": ["destination"],
    "phrases": [
        "How long to get to {destination}",
        "How long will it take to drive to {destination}",
        "How long is the drive to {destination}",
        "Time to get to {destination}",
        "How long from {origin} to {destination}",
        "How long to {destination} by {mode}",
        "What's the travel time to {destination}",
        "How many minutes to {destination}",
        "How long until I get to {destination}",
        "Estimated time to {destination}",
    ],
    "param_values": {
        "destination": DESTINATIONS,
        "origin": ["home", "work", "here", "the airport"],
        "mode": TRAVEL_MODES,
        "time": DATES,
    },
}

TOOLS["nav_eta"] = {
    "domain": "navigation",
    "description": "Get estimated arrival or departure time",
    "parameters": {
        "destination": {"type": "string"},
        "origin": {"type": "string"},
        "time": {"type": "string"},
        "type": {"type": "string", "enum": ["arrival", "departure"]},
        "mode": {"type": "string"},
    },
    "required": ["destination"],
    "phrases": [
        ("What time will I arrive at {destination}", {"type": "arrival"}),
        ("When will I get to {destination}", {"type": "arrival"}),
        ("ETA to {destination}", {"type": "arrival"}),
        ("What's my ETA to {destination}", {"type": "arrival"}),
        ("When should I leave for {destination}", {"type": "departure"}),
        ("What time should I leave to get to {destination} by {time}", {"type": "departure"}),
        ("When do I need to leave for {destination}", {"type": "departure"}),
        ("What time do I need to depart for {destination}", {"type": "departure"}),
        ("When should I head out to {destination}", {"type": "departure"}),
    ],
    "param_values": {
        "destination": DESTINATIONS,
        "origin": ["home", "work", "here"],
        "time": TIMES,
        "type": ["arrival", "departure"],
        "mode": TRAVEL_MODES,
    },
}

TOOLS["nav_distance"] = {
    "domain": "navigation",
    "description": "Get distance between locations",
    "parameters": {
        "destination": {"type": "string"},
        "origin": {"type": "string"},
        "unit": {"type": "string"},
        "mode": {"type": "string"},
    },
    "required": ["destination"],
    "phrases": [
        "How far is {destination}",
        "How far is it to {destination}",
        "Distance to {destination}",
        "How many miles to {destination}",
        "How far from {origin} to {destination}",
        "What's the distance to {destination}",
        "How far away is {destination}",
        "Miles to {destination}",
    ],
    "param_values": {
        "destination": DESTINATIONS,
        "origin": ["home", "work", "here", "the airport", "downtown"],
        "unit": ["miles", "kilometers"],
        "mode": TRAVEL_MODES,
    },
}

# =============================================================================
# DOMAIN: EVENT (1 function)
# =============================================================================

TOOLS["event_query"] = {
    "domain": "event",
    "description": "Search for events",
    "parameters": {
        "name": {"type": "string"},
        "time": {"type": "string"},
        "location": {"type": "string"},
        "category": {"type": "string"},
    },
    "required": [],
    "phrases": [
        "What events are happening {date}",
        "Find {category} events near {location}",
        "Any {category} events {date}",
        "What's happening {date}",
        "Events near me",
        "Find {name} near {location}",
        "Are there any {name} {date}",
        "What {category} events are coming up",
        "Search for {name} in {location}",
        "What's going on this weekend",
        "Events in {location}",
        "Any {name} near me",
    ],
    "param_values": {
        "name": EVENT_NAMES,
        "time": TIMES,
        "date": DATES,
        "location": LOCATIONS,
        "category": EVENT_CATEGORIES,
    },
}

# -----------------------------------------------------------------------------
# TOOL SCHEMA GENERATION
# -----------------------------------------------------------------------------

def generate_tool_schema() -> dict:
    """Generate the complete tool schema for the model."""
    functions = []
    for name, config in TOOLS.items():
        func = {
            "name": name,
            "description": config["description"],
            "parameters": {
                "type": "object",
                "properties": config["parameters"],
                "required": config["required"],
            },
        }
        functions.append(func)
    return {"functions": functions}


# -----------------------------------------------------------------------------
# SAMPLE GENERATION
# -----------------------------------------------------------------------------

@dataclass
class Sample:
    transcript: str
    tool_call: dict
    voice: str
    split: str
    domain: str


def generate_sample(split: str) -> Sample:
    """Generate a single sample."""
    tool_name = random.choice(list(TOOLS.keys()))
    tool_config = TOOLS[tool_name]

    # Parse phrase entry (tuple with fixed params, or plain string)
    phrase_entry = random.choice(tool_config["phrases"])
    if isinstance(phrase_entry, tuple):
        phrase_template, fixed_params = phrase_entry
    else:
        phrase_template, fixed_params = phrase_entry, {}

    arguments = {}
    arguments.update(fixed_params)  # Apply phrase-specific bindings first

    for param_name, param_values in tool_config.get("param_values", {}).items():
        if param_name in arguments:
            continue  # Already fixed by phrase binding
        placeholder = "{" + param_name + "}"
        if placeholder in phrase_template or (param_name in tool_config["required"]):
            arguments[param_name] = random.choice(param_values)

    transcript = phrase_template
    for param_name, param_value in arguments.items():
        placeholder = "{" + param_name + "}"
        if placeholder in transcript:
            display_value = str(param_value).replace("_", " ")
            transcript = transcript.replace(placeholder, display_value)

    transcript = re.sub(r'\{[^}]+\}', '', transcript).strip()
    transcript = re.sub(r'\s+', ' ', transcript)

    tool_call = {
        "function": tool_name,
        "arguments": arguments if arguments else {},
    }

    voice = random.choice(VOICES[split])

    return Sample(
        transcript=transcript,
        tool_call=tool_call,
        voice=voice,
        split=split,
        domain=tool_config["domain"],
    )


def generate_samples(num_samples: int, split_ratios: tuple = (0.8, 0.1, 0.1)) -> list[Sample]:
    """Generate samples with train/val/test splits."""
    samples = []

    train_count = int(num_samples * split_ratios[0])
    val_count = int(num_samples * split_ratios[1])
    test_count = num_samples - train_count - val_count

    for _ in range(train_count):
        samples.append(generate_sample("train"))
    for _ in range(val_count):
        samples.append(generate_sample("val"))
    for _ in range(test_count):
        samples.append(generate_sample("test"))

    return samples


# -----------------------------------------------------------------------------
# AUDIO GENERATION
# -----------------------------------------------------------------------------

async def generate_audio(text: str, voice: str, output_path: str, max_retries: int = 3):
    """Generate audio file using Edge TTS with retry on transient failures."""
    import edge_tts
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise


async def generate_dataset(output_dir: str, num_samples: int = 50, workers: int = 5):
    """Generate the full dataset with audio files. Resumes from checkpoint if interrupted."""
    from asyncio import Semaphore, Lock

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (output_path / split / "audio").mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} samples...")
    samples = generate_samples(num_samples)

    # Count domains
    domain_counts = {}
    for s in samples:
        domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1
    print(f"Domain distribution: {domain_counts}")

    schema = generate_tool_schema()
    with open(output_path / "tool_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Saved tool schema with {len(schema['functions'])} functions")

    # --- Checkpoint: load already-completed samples ---
    checkpoint_path = output_path / "checkpoint.jsonl"
    done_indices = set()
    metadata = {"train": [], "val": [], "test": []}

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                idx = entry.pop("_idx")
                done_indices.add(idx)
                split = entry["audio_path"].split("/")[0]
                metadata[split].append(entry)
        print(f"  Resuming: {len(done_indices)} samples already done, {num_samples - len(done_indices)} remaining")

    sem = Semaphore(workers)
    ckpt_lock = Lock()

    async def process_sample(idx: int, sample: Sample):
        async with sem:
            filename = f"sample_{idx:05d}.mp3"
            audio_path = output_path / sample.split / "audio" / filename

            await generate_audio(sample.transcript, sample.voice, str(audio_path))

            rel_audio_path = f"{sample.split}/audio/{filename}"
            entry = {
                "audio_path": rel_audio_path,
                "transcript": sample.transcript,
                "tool_call": sample.tool_call,
                "voice": sample.voice,
                "domain": sample.domain,
            }

            async with ckpt_lock:
                metadata[sample.split].append(entry)
                with open(checkpoint_path, "a") as f:
                    f.write(json.dumps({"_idx": idx, **entry}) + "\n")

            return idx

    tasks = [process_sample(i, sample) for i, sample in enumerate(samples)
             if i not in done_indices]

    completed = len(done_indices)
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        if completed % 50 == 0 or completed == num_samples:
            print(f"  Generated {completed}/{num_samples}")

    for split in ["train", "val", "test"]:
        metadata_path = output_path / split / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for entry in metadata[split]:
                f.write(json.dumps(entry) + "\n")
        print(f"  {split}: {len(metadata[split])} samples")

    print(f"\nDataset saved to {output_dir}")


# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic voice command dataset")
    parser.add_argument("--output", "-o", default="./data/synthetic", help="Output directory")
    parser.add_argument("--samples", "-n", type=int, default=50, help="Number of samples")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Concurrent TTS workers")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    asyncio.run(generate_dataset(args.output, args.samples, args.workers))
