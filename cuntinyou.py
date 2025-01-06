"""
Detta skript implementerar ett system för kontinuerlig kodförbättring. Det söker igenom specifika kataloger efter filer med angivna filändelser, väljer slumpmässigt en fil och använder Google Gemini API för att föreslå förbättringar av koden. Förbättringarna skrivs tillbaka till filen om de anses vara fördelaktiga. Skriptet hanterar även loggning och felhantering för att säkerställa robust drift.
"""

import asyncio
import configparser
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Ladda .env-filen för att hämta API-nyckeln
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ladda konfigurationsfilen
config = configparser.ConfigParser()
config.read("config.ini")

# Läs inställningar från config
FILE_EXTENSIONS = config["FILES"]["extensions"].split(",")
SRC_DIR = config["FILES"]["src_dir"]
LOG_LEVEL = config["LOGGING"]["level"]
LOG_FILE = config["LOGGING"]["log_file"]

# Ställ in klient för Google Gemini
client = genai.PromptServiceClient(credentials=GEMINI_API_KEY)

# Logger för bättre felsökning
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
logger.addHandler(file_handler)


class ContinuousImprover:
    """System för kontinuerlig kodförbättring."""

    def __init__(self):
        """Initierar ContinuousImprover med inställningar från configfilen."""
        self.results_file = config["FILES"]["results_file"]
        self.results = {}
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Ställer in loggning för ContinuousImprover.

        Returns:
            logging.Logger: Logger för ContinuousImprover.
        """
        logger = logging.getLogger("ContinuousImprover")
        logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
        logger.addHandler(file_handler)
        return logger

    async def improve_random_file(self):
        """Väljer och förbättrar en slumpmässig fil."""
        files = self.get_improvable_files()
        if not files:
            self.logger.warning("Inga filer hittades att förbättra.")
            return
        file_path = random.choice(files)
        await self.improve_file(file_path)

    def get_improvable_files(self) -> List[Path]:
        """Hämtar en lista på filer som kan förbättras.

        Returns:
            List[Path]: Lista över sökvägar till filer som kan förbättras.
        """
        files = []
        for extension in FILE_EXTENSIONS:
            files.extend(Path(SRC_DIR).rglob(f"*{extension}"))
        return files

    async def improve_file(self, file_path: Path):
        """Förbättrar en enskild fil.

        Args:
            file_path (Path): Sökvägen till filen som ska förbättras.
        """
        try:
            self.logger.info(f"Försöker förbättra {file_path}")
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()

            backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
            async with aiofiles.open(backup_path, "w", encoding="utf-8") as f:
                await f.write(content)

            try:
                improved_content = await self.get_improvements(content)

                if improved_content and improved_content != content:
                    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                        await f.write(improved_content)
                    result = ImprovementResult(
                        file_path=str(file_path),
                        success=True,
                        changes_made=True,
                        timestamp=datetime.now(),
                    )
                    self.logger.info(f"Framgångsrikt förbättrat {file_path}")
                else:
                    result = ImprovementResult(
                        file_path=str(file_path),
                        success=True,
                        changes_made=False,
                        timestamp=datetime.now(),
                    )
                    self.logger.info(f"Inga förbättringar behövdes för {file_path}")

            except Exception as e:
                if backup_path.exists():
                    await aiofiles.os.replace(backup_path, file_path)
                result = ImprovementResult(
                    file_path=str(file_path),
                    success=False,
                    changes_made=False,
                    timestamp=datetime.now(),
                    error=str(e),
                )
                self.logger.error(
                    f"Misslyckades med att förbättra {file_path}: {str(e)}"
                )

            if str(file_path) not in self.results:
                self.results[str(file_path)] = []
            self.results[str(file_path)].append(result)
            if backup_path.exists():
                await aiofiles.os.remove(backup_path)

        except Exception as e:
            self.logger.error(f"Fel vid bearbetning av {file_path}: {str(e)}")

    async def get_improvements(self, content: str) -> Optional[str]:
        """Hämtar förbättringar från Gemini API.

        Args:
            content (str): Koden som ska förbättras.

        Returns:
            Optional[str]: Förbättrad kod eller originalkoden om förbättring misslyckas.
        """
        try:
            prompt = types.Prompt(
                text=f"Förbättra följande kod:\n{content}\n\nVänligen förbättra denna kod för att:\n"
                "1. Höja kodens kvalitet.\n"
                "2. Förbättra prestanda.\n"
                "3. Lägg till bättre felhantering.\n"
                "4. Förbättra dokumentationen.\n"
                "5. Följ bästa praxis."
            )
            response = await client.generate_text(prompt=prompt)

            if response.choices:
                improved_code = response.choices[0].text.strip()
                return improved_code
            self.logger.error("Ingen förbättring returnerades från Gemini.")
            return content
        except Exception as e:
            self.logger.error(f"Fel vid förbättring av kod: {str(e)}")
            return content


class ImprovementResult:
    """Lagrar resultatet av ett försök till kodförbättring."""

    def __init__(
        self,
        file_path: str,
        success: bool,
        changes_made: bool,
        timestamp: datetime,
        error: Optional[str] = None,
    ):
        """Initierar ett resultatobjekt för kodförbättring.

        Args:
            file_path (str): Sökvägen till filen som bearbetades.
            success (bool): Om förbättringen lyckades eller inte.
            changes_made (bool): Om ändringar gjordes i filen.
            timestamp (datetime): Tidpunkt då förbättringen gjordes.
            error (Optional[str], valfritt): Felmeddelande om förbättringen misslyckades.
        """
        self.file_path = file_path
        self.success = success
        self.changes_made = changes_made
        self.timestamp = timestamp
        self.error = error

    def __repr__(self):
        """Representation av ImprovementResult för loggning och felsökning."""
        return (
            f"ImprovementResult(file_path={self.file_path}, success={self.success}, "
            f"changes_made={self.changes_made}, timestamp={self.timestamp}, error={self.error})"
        )


async def main():
    """Huvudfunktion som initierar ContinuousImprover och förbättrar en slumpmässig fil."""
    improver = ContinuousImprover()
    await improver.improve_random_file()


if __name__ == "__main__":
    asyncio.run(main())
