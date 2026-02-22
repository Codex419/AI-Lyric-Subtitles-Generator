from textblob import TextBlob

MOOD_LIST = [
    "Abstract", "Accepting", "Adventurous", "Aggressive", "Alluring", "Angry", "Anguished", "Animated", "Anticipating",
    "Anxious", "Apprehensive", "Atmospheric", "Beautiful", "Bitchy", "Bitter", "Bittersweet", "Bizarre", "Bold",
    "Booming", "Bouncy", "Brave", "Bright", "Building", "Buoyant", "Busy", "Calm", "Campy", "Carefree", "Careful",
    "Caring", "Cautious", "Celebratory", "Chaotic", "Cheeky", "Cheerful", "Childish", "Climactic", "Cold", "Comforting",
    "Confident", "Confrontational", "Confused", "Contemplative", "Cool", "Crazy", "Creepy", "Crunk", "Curious",
    "Dancing", "Dangerous", "Dark", "Deep", "Defeated", "Delicate", "Delighted", "Depressed", "Determined",
    "Disenchanted", "Disillusioned", "Dissonant", "Disturbing", "Doubtful", "Dramatic", "Dreadful", "Dreamy",
    "Driving", "Droning", "Drunk", "Dynamic", "Earthy", "Easy", "Ecstatic", "Edgy", "Eerie", "Elated", "Elegant",
    "Emotional", "Enchanted", "Energetic", "Epic", "Erotic", "Escalating", "Ethereal", "Evil", "Excited", "Exotic",
    "Explosive", "Fast", "Fearful", "Festive", "Fiery", "Flirtatious", "Flowing", "Forceful", "Foreboding", "Forgiving",
    "Frantic", "Freaky", "Free", "Fresh", "Friendly", "Frisky", "Fun", "Funky", "Funny", "Glorious", "Good", "Graceful",
    "Grand", "Greasy", "Gritty", "Groovy", "Gutsy", "Happy", "Hard", "Haunting", "Heartbroken", "Heartening",
    "Heartwarming", "Heated", "Heavenly", "Heavy", "Hectic", "Helpless", "Heroic", "Honest", "Hopeful", "Hopeless",
    "Horny", "Hot", "Humble", "Hurt", "Hypnotic", "Innocent", "Inquisitive", "Inspiring", "Intelligent", "Intense",
    "Intricate", "Jangly", "Jealous", "Jolly", "Joyful", "Jumpy", "Kind", "Light", "Lively", "Lofty", "Lonely",
    "Longing", "Lost", "Loud", "Loving", "Mad", "Magical", "Majestic", "Marching", "Mean", "Meaningful", "Mechanical",
    "Meditative", "Melancholic", "Mellow", "Menacing", "Mischievous", "Moody", "Motivational", "Mournful", "Moving",
    "Mysterious", "Mystical", "Nervous", "Noble", "Nostalgic", "Obsessive", "Ominous", "Oppressed", "Optimistic",
    "Outgoing", "Painful", "Passionate", "Patriotic", "Peaceful", "Pensive", "Playful", "Pleading", "Pleased",
    "Pompous", "Positive", "Powerful", "Primitive", "Proud", "Pulsating", "Purposeful", "Pushy", "Questioning", "Quiet",
    "Quirky", "Rambling", "Raunchy", "Raw", "Rebellious", "Reflective", "Regal", "Regretful", "Relaxing", "Repetitive",
    "Restless", "Retro", "Reverent", "Rhythmic", "Risque", "Romantic", "Rousing", "Sad", "Scared", "Scary", "Schmaltzy",
    "Secretive", "Sedate", "Sensitive", "Sensual", "Sentimental", "Serene", "Serious", "Sexy", "Shimmering", "Sick",
    "Silly", "Simple", "Sincere", "Sinister", "Slow", "Smokey", "Smooth", "Sneaky", "Snobbish", "Soaring", "Soft",
    "Solemn", "Somber", "Soothing", "Sophisticated", "Sorry", "Soulful", "Spacious", "Sparse", "Spirited", "Sprightly",
    "Stable", "Stately", "Stimulating", "Stirring", "Strange", "Street-Smart", "Striking", "Strong", "Sublime", "Subtle",
    "Successful", "Suffocated", "Suggestive", "Summery", "Surprising", "Suspenseful", "Suspicious", "Swaggering",
    "Sweeping", "Sweet", "Swinging", "Swirling", "Tender", "Tension", "Terror", "Thankful", "Thinking", "Thoughtful",
    "Thrilling", "Touching", "Tough", "Tragic", "Trance", "Tranquil", "Triumphant", "Ugly", "Unfriendly", "Uplifting",
    "Vengeful", "Vibrant", "Wacky", "Warm", "Whimsical", "Wholesome", "Wicked", "Wild", "Wondrous", "Worried", "Wrong"
]

class MoodAnalyzer:
    @staticmethod
    def get_moods_from_text(text):
        """Analyzes text to extract 3-7 moods from MOOD_LIST."""
        if not text: return ["Unknown", "Neutral", "Quiet"] # Fallback

        text_lower = text.lower()
        found_moods = set()

        # 1. Direct Keyword Matching
        for mood in MOOD_LIST:
            if mood.lower() in text_lower:
                found_moods.add(mood)

        # 2. Sentiment Analysis Fallback
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity # -1 to 1
            subjectivity = blob.sentiment.subjectivity # 0 to 1

            if sentiment > 0.5: found_moods.update(["Positive", "Happy", "Bright"])
            elif sentiment < -0.5: found_moods.update(["Negative", "Sad", "Dark"])
            elif sentiment > 0: found_moods.update(["Optimistic", "Light"])
            elif sentiment < 0: found_moods.update(["Melancholic", "Serious"])
            else: found_moods.update(["Neutral", "Calm"])

            if subjectivity > 0.7: found_moods.update(["Emotional", "Passionate"])
            elif subjectivity < 0.3: found_moods.update(["Simple", "Direct"])

        except Exception: pass # TextBlob might fail if corpora missing

        # 3. Ensure 3-7 Moods
        final_moods = list(found_moods)

        # Pad if < 3
        if len(final_moods) < 3:
            # Add generics if not already present
            defaults = ["Atmospheric", "Reflective", "Narrative", "Musical", "Rhythmic", "Vocal"]
            for d in defaults:
                if d not in final_moods:
                    final_moods.append(d)
                    if len(final_moods) >= 3: break

        # Truncate if > 7 (random sample or just first N? First N usually better for reproducibility)
        if len(final_moods) > 7:
            final_moods = final_moods[:7]

        return final_moods
