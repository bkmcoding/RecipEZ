from ingredient_parser import parse_ingredient

# Example data
raw_data = "3 ¾ cups cubed white bread, 1 ½ cups cubed whole wheat bread, 1 pound ground turkey sausage, 1 cup chopped onion, ¾ cup chopped celery, 2 ½ teaspoons dried sage, 1 ½ teaspoons dried rosemary, ½ teaspoon dried thyme, 1 Golden Delicious apple, cored and chopped, ¾ cup dried cranberries, ⅓ cup minced fresh parsley, 1 cooked turkey liver, finely chopped, ¾ cup turkey stock, 4 tablespoons unsalted butter, melted"

phrases = [phrase.strip() for phrase in raw_data.split(',')]

cleaned_ingredients = []

for phrase in phrases:
    if not phrase: 
        continue
    parsed = parse_ingredient(phrase)
    
    # Keeps only name attributte
    if parsed.name:
        core_food = parsed.name[0].text 
        
        # Replaces spaces with underscores for vectors
        formatted_food = core_food.replace(" ", "_")
        cleaned_ingredients.append(formatted_food)

final_output = " ".join(cleaned_ingredients)

print(final_output)