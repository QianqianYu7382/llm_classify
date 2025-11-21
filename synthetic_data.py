

import re
import random
from typing import List, Tuple

TOPIC_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def generate_synthetic_news(
    n_per_class: int = 500,
    random_state: int = 42,
) -> Tuple[List[str], List[int]]:

    random.seed(random_state)


    templates = {
        0: [  # World
            "leaders discuss {issue} in {region} summit",
            "{country} election results spark {reaction}",
            "protests erupt in {city} over {issue}",
        ],
        1: [  # Sports
            "{team1} defeat {team2} in {score} victory",
            "{player} sets new {sport} record",
            "{team1} coach praises defense after {score} win",
        ],
        2: [  # Business
            "{company} shares rise after {event}",
            "central bank cuts interest rates amid {condition}",
            "{company} announces merger with {company2}",
        ],
        3: [  # Sci/Tech
            "scientists discover new {object} in space",
            "{company} launches new {device} with {feature}",
            "researchers develop {tech} to improve {field}",
        ],
    }

    slot_values = {
        "issue": [
            "climate change",
            "trade dispute",
            "security concerns",
            "border tensions",
        ],
        "region": ["europe", "asia", "africa", "middle east"],
        "country": ["us", "china", "india", "france", "brazil"],
        "reaction": ["celebrations", "fears", "protests", "uncertainty"],
        "city": ["paris", "london", "tokyo", "berlin"],

        "team1": ["lakers", "yankees", "barcelona", "warriors"],
        "team2": ["celtics", "dodgers", "real madrid", "heat"],
        "score": ["3-1", "2-0", "5-4", "1-0"],
        "player": ["james", "ronaldo", "messi", "federer"],
        "sport": ["tennis", "football", "basketball", "baseball"],

        "company": ["apple", "google", "amazon", "tesla", "microsoft"],
        "company2": ["startupx", "fintechy", "biotechz"],
        "event": ["earnings beat", "strong sales", "regulatory approval"],
        "condition": ["recession fears", "slow growth", "inflation concerns"],

        "object": ["planet", "galaxy", "asteroid", "black hole"],
        "device": ["smartphone", "chip", "drone", "robot"],
        "feature": ["longer battery life", "faster processing", "ai assistant"],
        "tech": ["algorithm", "sensor", "material", "software"],
        "field": ["healthcare", "transportation", "energy", "agriculture"],
    }

    def fill_template(tpl: str) -> str:
        """
        用 slot_values 替换模板中的 {xxx} 占位符，生成一条具体句子。
        """
        while True:
            m = re.search(r"{(.*?)}", tpl)
            if not m:
                break
            key = m.group(1)
            choices = slot_values.get(key, ["value"])
            val = random.choice(choices)
            tpl = tpl.replace("{" + key + "}", val, 1)
        return tpl

    texts: List[str] = []
    labels: List[int] = []

    for label in range(4):
        tpls = templates[label]
        for _ in range(n_per_class):
            tpl = random.choice(tpls)
            text = fill_template(tpl)
            texts.append(text)
            labels.append(label)

    return texts, labels


if __name__ == "__main__":
    texts, labels = generate_synthetic_news(n_per_class=5, random_state=0)
    for t, y in zip(texts, labels):
        print(f"[{y} - {TOPIC_NAMES[y]}] {t}")
