#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyserar Python-filer för att identifiera funktioner som kan dra nytta av dekoratorer.
Visar resultatet i ett ncurses-baserat gränssnitt liknande ncdu.
"""

import ast
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging
import aiofiles
import curses
from curses import wrapper
from statistics import mean

# Konfigurera loggning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_function(node: ast.FunctionDef) -> Dict[str, Any]:
    """
    Analyserar en funktion för att avgöra hur mycket den skulle dra nytta av en dekorator.

    Args:
        node (ast.FunctionDef): Funktionens AST-nod.

    Returns:
        Dict[str, Any]: En dictionary med analysresultat.
    """
    decorators = len(node.decorator_list)
    has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
    cyclomatic_complexity = calculate_cyclomatic_complexity(node)
    nested_functions = sum(
        1 for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n != node
    )
    length = len(node.body)

    metrics = {
        'decorators': decorators,
        'has_return': has_return,
        'cyclomatic_complexity': cyclomatic_complexity,
        'nested_functions': nested_functions,
        'length': length,
    }
    return metrics

def calculate_cyclomatic_complexity(node: ast.AST) -> int:
    """
    Beräknar den cyklomatiska komplexiteten för en funktion.

    Args:
        node (ast.AST): Funktionens AST-nod.

    Returns:
        int: Den cyklomatiska komplexiteten.
    """
    complexity = 1
    for n in ast.walk(node):
        if isinstance(
            n,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.Try,
                ast.With,
                ast.And,
                ast.Or,
                ast.ExceptHandler,
                ast.BoolOp,
            ),
        ):
            complexity += 1
    return complexity

def evaluate_function(metrics: Dict[str, Any]) -> int:
    """
    Utvärderar en funktions metrik för att ge en poäng för fördelen med att lägga till dekoratorer.

    Args:
        metrics (Dict[str, Any]): Funktionens metrik.

    Returns:
        int: Poäng mellan 0 och 100.
    """
    score = 0
    if metrics['decorators'] == 0:
        score += 10
    if metrics['has_return']:
        score += 15
    score += metrics['cyclomatic_complexity'] * 2
    score += metrics['nested_functions'] * 5
    score += min(metrics['length'] // 5, 10)
    return min(score, 100)

async def analyze_file(
    file_path: Path, cache: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Analyserar en Python-fil för funktioner som kan dra nytta av dekoratorer.

    Args:
        file_path (Path): Sökvägen till filen.
        cache (Dict[str, List[Dict[str, Any]]]): Cache för tidigare analyserade filer.

    Returns:
        List[Dict[str, Any]]: Lista över analysresultat.
    """
    if str(file_path) in cache:
        return cache[str(file_path)]
    results = []
    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics = analyze_function(node)
                    score = evaluate_function(metrics)
                    results.append(
                        {
                            'function': node.name,
                            'score': score,
                            'file': str(file_path),
                            'metrics': metrics,
                        }
                    )
        cache[str(file_path)] = results
    except (SyntaxError, IOError, UnicodeDecodeError) as e:
        logging.error(f"Fel vid analys av {file_path}: {e}")
    except Exception as e:
        logging.exception(f"Ohanterat undantag vid analys av {file_path}: {e}")
    return results

async def analyze_directory(
    directory: Path, cache: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Analyserar alla Python-filer i en katalog för potentiella dekoratorfördelar, med caching.

    Args:
        directory (Path): Sökvägen till katalogen.
        cache (Dict[str, List[Dict[str, Any]]]): Cache för tidigare analyserade filer.

    Returns:
        List[Dict[str, Any]]: Platt lista över alla analysresultat.
    """
    tasks = []
    for file_path in directory.rglob("*.py"):
        tasks.append(analyze_file(file_path, cache))
    results = await asyncio.gather(*tasks)
    flat_results = [item for sublist in results for item in sublist]
    return flat_results

def print_summary(results: List[Dict[str, Any]]) -> None:
    """
    Skriver ut en sammanfattning av analysen.

    Args:
        results (List[Dict[str, Any]]): Lista över analysresultat.
    """
    total_functions = len(results)
    avg_score = mean(result['score'] for result in results) if results else 0
    print(f"\nAnalyserade {total_functions} funktioner.")
    print(f"Genomsnittligt dekoratorpoäng: {avg_score:.2f}")

def ncurses_display(stdscr, results: List[Dict[str, Any]]) -> None:
    """
    Visar resultaten i ett ncurses-baserat gränssnitt.

    Args:
        stdscr: Standard-skärmen från curses.
        results (List[Dict[str, Any]]): Lista över analysresultat.
    """
    curses.curs_set(0)
    stdscr.clear()
    k = 0
    selected_row = 0

    files = {}
    for result in results:
        file = result['file']
        if file not in files:
            files[file] = []
        files[file].append(result)

    file_list = list(files.keys())

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        title = "Analyserade Funktioner - Navigera med piltangenterna, q för att avsluta"
        stdscr.addstr(0, 0, title[:width])

        for idx, file in enumerate(file_list):
            x = 0
            y = idx + 2
            if y >= height - 1:
                break
            if idx == selected_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, file[:width])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, file[:width])

        stdscr.refresh()

        k = stdscr.getch()

        if k == ord('q'):
            break
        elif k == curses.KEY_UP and selected_row > 0:
            selected_row -= 1
        elif k == curses.KEY_DOWN and selected_row < len(file_list) - 1:
            selected_row += 1
        elif k in (curses.KEY_ENTER, 10, 13):
            show_file_functions(stdscr, files[file_list[selected_row]])

def show_file_functions(stdscr, functions: List[Dict[str, Any]]) -> None:
    """
    Visar funktioner inom en vald fil.

    Args:
        stdscr: Standard-skärmen från curses.
        functions (List[Dict[str, Any]]): Lista över funktioner i filen.
    """
    curses.curs_set(0)
    k = 0
    selected_row = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        title = f"Funktioner i {functions[0]['file']} - Backa med b"
        stdscr.addstr(0, 0, title[:width])

        for idx, func in enumerate(functions):
            x = 0
            y = idx + 2
            if y >= height - 1:
                break
            func_display = f"{func['function']} - Poäng: {func['score']}"
            if idx == selected_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, func_display[:width])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, func_display[:width])

        stdscr.refresh()

        k = stdscr.getch()

        if k == ord('b'):
            break
        elif k == curses.KEY_UP and selected_row > 0:
            selected_row -= 1
        elif k == curses.KEY_DOWN and selected_row < len(functions) - 1:
            selected_row += 1
        elif k in (curses.KEY_ENTER, 10, 13):
            show_function_details(stdscr, functions[selected_row])

def show_function_details(stdscr, func: Dict[str, Any]) -> None:
    """
    Visar detaljer om en vald funktion.

    Args:
        stdscr: Standard-skärmen från curses.
        func (Dict[str, Any]): Funktionens data.
    """
    curses.curs_set(0)
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    details = [
        f"Fil: {func['file']}",
        f"Funktion: {func['function']}",
        f"Poäng: {func['score']}",
        "Metrik:",
        f"  Antal dekoratorer: {func['metrics']['decorators']}",
        f"  Har return-sats: {func['metrics']['has_return']}",
        f"  Cyklomatisk komplexitet: {func['metrics']['cyclomatic_complexity']}",
        f"  Nästlade funktioner: {func['metrics']['nested_functions']}",
        f"  Antal rader: {func['metrics']['length']}",
    ]

    for idx, line in enumerate(details):
        y = idx + 2
        if y >= height - 1:
            break
        stdscr.addstr(y, 0, line[:width])

    stdscr.addstr(height - 1, 0, "Tryck på valfri tangent för att återgå")
    stdscr.refresh()
    stdscr.getch()

def main() -> None:
    """
    Huvudfunktionen som analyserar katalogen och startar ncurses-gränssnittet.
    """
    if len(sys.argv) != 2:
        print("Användning: python script_name.py KATALOG")
        sys.exit(1)
    directory_to_analyze = Path(sys.argv[1])

    cache_file = 'analysis_cache.json'
    cache = {}

    if os.path.exists(cache_file):
        with open(cache_file, mode='r', encoding='utf-8') as f:
            cache = json.load(f)

    results = asyncio.run(analyze_directory(directory_to_analyze, cache))

    with open(cache_file, mode='w', encoding='utf-8') as f:
        json.dump(cache, f)

    print_summary(results)

    curses.wrapper(run_curses_interface, results)

def run_curses_interface(stdscr, results: List[Dict[str, Any]]) -> None:
    """
    Startar det ncurses-baserade gränssnittet.

    Args:
        stdscr: Standard-skärmen från curses.
        results (List[Dict[str, Any]]): Lista över analysresultat.
    """
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    ncurses_display(stdscr, results)

if __name__ == "__main__":
    main()
