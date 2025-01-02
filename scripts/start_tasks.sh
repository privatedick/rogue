#!/bin/bash

# Starta systemhälsokontroll
python src/tools/system_health.py

# Om allt är OK, starta AI-assistenten i bakgrunden
if [ $? -eq 0 ]; then
    # Starta AI-assistenten i en ny terminal
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open -a Terminal.app python src/tools/code_modifier.py
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v gnome-terminal &> /dev/null; then
            gnome-terminal -- python src/tools/code_modifier.py
        elif command -v xterm &> /dev/null; then
            xterm -e python src/tools/code_modifier.py &
        else
            echo "Kunde inte hitta lämplig terminal-emulator"
        fi
    elif [[ "$OSTYPE" == "msys" ]]; then
        # Windows Git Bash
        start python src/tools/code_modifier.py
    else
        echo "Kunde inte identifiera operativsystem"
    fi
fi
