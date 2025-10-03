import os
import json
import random

def build_router_training_data(max_per_category=10):
    """
    Samlar träningsdata från datasets-mappar.
    
    Args:
        max_per_category (int): Max antal frågor att ta från varje kategori.
    """
    base_dir = "datasets"
    output_file = "routertrainingdata.json"

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Mappen '{base_dir}' finns inte.")

    all_items = []
    category_label = 0

    # Gå igenom alla undermappar i datasets/ (i alfabetisk ordning)
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        json_path = os.path.join(folder_path, "data", f"{folder_name}.json")
        if not os.path.exists(json_path):
            print(f"Varning: Saknar fil → {json_path}")
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Stöd både lista och enskilt objekt
            items = data if isinstance(data, list) else [data]
            valid_questions = []

            for item in items:
                q = item.get("instruction", "").strip()
                if q:
                    valid_questions.append({
                        "question": q,
                        "category_label": category_label,
                        "difficulty_label": 0,
                        "privacy_label": 0
                    })

            # Begränsa till max_per_category (slumpmässig urval om för många)
            if len(valid_questions) > max_per_category:
                selected = random.sample(valid_questions, max_per_category)
            else:
                selected = valid_questions

            all_items.extend(selected)
            print(f"✅ Kategori '{folder_name}' (label {category_label}): {len(selected)} frågor")
            category_label += 1

        except Exception as e:
            print(f"Fel vid läsning av {json_path}: {e}")

    # Spara till fil
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Klart! Totalt {len(all_items)} frågor sparade till '{output_file}'")
    print(f"Antal kategorier: {category_label}")

# === KÖR SKRIPTET ===
if __name__ == "__main__":
    # 🔢 ÄNDRA DETTA VÄRDE FÖR ATT STYRA MAX ANTAL PER KATEGORI
    MAX_PER_CATEGORY = 400  # ←←← STÄLL IN HÄR

    build_router_training_data(max_per_category=MAX_PER_CATEGORY)