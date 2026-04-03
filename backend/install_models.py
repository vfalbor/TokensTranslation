#!/usr/bin/env python3
"""
install_models.py — Download argostranslate language models for local translation.

VPS specs: 8 GB RAM, 4 cores, 240 GB storage
Each model pair: ~300-400 MB disk, ~350 MB RAM when loaded.
This script installs the recommended set for a typical multilingual deployment.

Usage:
    python3 install_models.py              # Install recommended set
    python3 install_models.py --all        # Install all available pairs
    python3 install_models.py --list       # List available packages
    python3 install_models.py es en fr de  # Install specific src languages → EN
"""

import argparse
import sys

try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:
    print("ERROR: argostranslate not installed.")
    print("Run: pip install argostranslate")
    sys.exit(1)

# ── Recommended pairs for an 8 GB VPS ──────────────────────────────────────
# Each pair listed as (from_code, to_code).
# We install X→EN and EN→X for each language to enable bidirectional translation.
# RAM budget: 8 pairs × ~350 MB = ~2.8 GB — leaves plenty for the OS and API.
RECOMMENDED = [
    ("es", "en"), ("en", "es"),   # Spanish
    ("fr", "en"), ("en", "fr"),   # French
    ("de", "en"), ("en", "de"),   # German
    ("pt", "en"), ("en", "pt"),   # Portuguese
    ("it", "en"), ("en", "it"),   # Italian
    ("ru", "en"), ("en", "ru"),   # Russian
    ("zh", "en"), ("en", "zh"),   # Chinese
    ("ar", "en"), ("en", "ar"),   # Arabic
]

# Full set — only use with 16 GB+ RAM or if loading models on demand
FULL_SET_EXTRA = [
    ("ja", "en"), ("en", "ja"),   # Japanese
    ("ko", "en"), ("en", "ko"),   # Korean
    ("nl", "en"), ("en", "nl"),   # Dutch
    ("pl", "en"), ("en", "pl"),   # Polish
    ("tr", "en"), ("en", "tr"),   # Turkish
    ("uk", "en"), ("en", "uk"),   # Ukrainian
    ("hi", "en"), ("en", "hi"),   # Hindi
]


def update_index():
    print("→ Updating package index...")
    argostranslate.package.update_package_index()
    print("  Index updated.\n")


def get_available():
    return argostranslate.package.get_available_packages()


def install_pair(from_code: str, to_code: str, available: list) -> bool:
    pkg = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None
    )
    if not pkg:
        print(f"  ✗ No package found for {from_code}→{to_code}")
        return False

    print(f"  ↓ Downloading {from_code}→{to_code} ({pkg.from_name} → {pkg.to_name})...")
    try:
        path = pkg.download()
        argostranslate.package.install_from_path(path)
        print(f"  ✓ Installed {from_code}→{to_code}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to install {from_code}→{to_code}: {e}")
        return False


def list_installed():
    installed = argostranslate.translate.get_installed_languages()
    if not installed:
        print("No languages installed yet.")
        return
    print(f"\nInstalled languages ({len(installed)}):")
    for lang in installed:
        translations = [t.to_lang.name for t in lang.translations_from]
        if translations:
            print(f"  {lang.code:6} {lang.name:20} → {', '.join(translations)}")


def main():
    parser = argparse.ArgumentParser(description="Install argostranslate models for TokenTranslation")
    parser.add_argument("langs", nargs="*", help="Specific source language codes to install (→EN direction)")
    parser.add_argument("--all", action="store_true", help="Install all language pairs (16 GB+ RAM recommended)")
    parser.add_argument("--list", action="store_true", help="List installed languages and exit")
    args = parser.parse_args()

    if args.list:
        list_installed()
        return

    print("╔══════════════════════════════════════════════╗")
    print("║  TokenTranslation — Model Installer           ║")
    print("╚══════════════════════════════════════════════╝\n")

    update_index()
    available = get_available()
    print(f"  {len(available)} packages available in registry.\n")

    # Determine which pairs to install
    if args.langs:
        pairs = []
        for code in args.langs:
            pairs.append((code, "en"))
            pairs.append(("en", code))
    elif args.all:
        pairs = RECOMMENDED + FULL_SET_EXTRA
    else:
        pairs = RECOMMENDED

    print(f"Installing {len(pairs)} translation pairs...\n")
    ok = 0
    for from_code, to_code in pairs:
        if install_pair(from_code, to_code, available):
            ok += 1

    print(f"\n{'='*48}")
    print(f"Done. {ok}/{len(pairs)} pairs installed successfully.")
    print()
    list_installed()
    print("\nRestart the TokenTranslation service to load new models:")
    print("  sudo systemctl restart tokentranslation")


if __name__ == "__main__":
    main()
