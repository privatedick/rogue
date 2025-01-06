#!/bin/bash

# Skapa en lista över alla Python-filer
echo "Skapar en lista över alla Python-filer..."
PY_FILES=$(find . -name "*.py")

# Lista för att hålla reda på filer med felaktiga importer
declare -a FILES_WITH_OLD_IMPORTS=()

# Sök efter gamla importer och lista dem
echo "Söker efter gamla importer..."
for FILE in $PY_FILES; do
  if grep -q 'from project_setup\|import project_setup' "$FILE"; then
    echo "Hittade felaktiga importer i: $FILE"
    FILES_WITH_OLD_IMPORTS+=("$FILE")
  fi
done

# Lista alla filer med felaktiga importer
if [ ${#FILES_WITH_OLD_IMPORTS[@]} -eq 0 ]; then
  echo "Inga felaktiga importer hittades."
else
  echo "Filer med felaktiga importer:"
  for FILE in "${FILES_WITH_OLD_IMPORTS[@]}"; do
    echo "$FILE"
  done
fi
