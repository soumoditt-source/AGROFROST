#!/usr/bin/env python3
"""
==========================================
EcoDrone AI - Raw Data Intelligence Core
Batch Processing Utility (v2.0)
Built for Kshitij 2026
==========================================
"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))
from backend.app.ml.pit_detector import detect_pits
from backend.app.ml.classifier import analyze_survival_at_pits
from backend.app.ml.registration import register_images

console = Console()

def process_folder(folder_path, gsd=2.5, use_gemini=False):
    folder = Path(folder_path)
    console.print(Panel(f"[bold green]Scanning Folder:[/bold green] {folder.name}", border_style="green"))
    
    # 1. Identify Files (Smart Search)
    ortho_dir = folder / "Ortho"
    if not ortho_dir.exists():
        ortho_dir = folder / "Ortho Data"
    
    if not ortho_dir.exists():
        console.print("[red]Error: Could not find 'Ortho' or 'Ortho Data' directories.[/red]")
        return
    
    op1_path = next(ortho_dir.rglob("*Pre-Pitting*.tif"), None)
    op3_path = next(ortho_dir.rglob("*Post-SW*.tif"), None) or next(ortho_dir.rglob("*Post-Pitting*.tif"), None)
    
    if not op1_path or not op3_path:
        console.print(f"[yellow]Warning: Missing OP1 or OP3 in {folder.name}. Skipping...[/yellow]")
        return

    console.print(f"  [cyan]OP1 (Reference):[/cyan] {op1_path.name}")
    console.print(f"  [cyan]OP3 (Analysis):[/cyan] {op3_path.name}")

    # 2. Pipeline Execution
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Load Data
        task_load = progress.add_task("[yellow]Loading high-res TIFs...", total=100)
        img1_bytes = op1_path.read_bytes()
        img3_bytes = op3_path.read_bytes()
        progress.update(task_load, completed=100)
        
        # Registration
        task_reg = progress.add_task("[blue]Aligning Temporal Data...", total=100)
        # Using the actual logic from registration.py
        registered_img3 = register_images(img1_bytes, img3_bytes)
        progress.update(task_reg, completed=100)
        
        # Detection
        task_det = progress.add_task("[green]Identifying Sample Pits...", total=100)
        pits = detect_pits(img1_bytes, gsd_cm_px=gsd)
        progress.update(task_det, completed=100)
        
        # Classification
        task_class = progress.add_task("[magenta]Evaluating Bio-Vitality...", total=len(pits))
        # Note: analyze_survival handles the batching/mapping internally
        stats = analyze_survival_at_pits(
            registered_img3 if registered_img3 is not None else img3_bytes,
            pits,
            gsd_cm_px=gsd,
            use_gemini=use_gemini
        )
        progress.update(task_class, completed=len(pits))

    # 3. Report Generation
    _display_summary(folder.name, stats)
    
    # Save Results
    output_meta = {
        "folder": folder.name,
        "metrics": stats,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    out_file = folder / f"analysis_report_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(output_meta, f, indent=4)
    
    console.print(f"\n[bold green]Success![/bold green] Report saved to: [underline]{out_file}[/underline]\n")

def _display_summary(name, stats):
    table = Table(title=f"Analysis Summary: {name}", box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")
    
    table.add_row("Total Pits Audited", str(stats['total']))
    table.add_row("Live Saplings Detected", str(stats['total'] - stats['dead']))
    table.add_row("Casualties Identified", str(stats['dead']))
    
    rate = stats['rate']
    color = "green" if rate > 80 else "yellow" if rate > 60 else "red"
    table.add_row("Aggregated Survival Rate", f"[{color}]{rate:.2f}%[/{color}]")
    
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoDrone AI Raw Data Processor")
    parser.add_argument("--root", default="Drone image", help="Root folder containing VFs")
    parser.add_argument("--gsd", type=float, default=2.5, help="Ground Sampling Distance (cm/px)")
    parser.add_argument("--gemini", action="store_true", help="Enable Gemini 1.5 Pro for ambiguous pits")
    
    args = parser.parse_args()
    
    console.print("\n[bold]EcoDrone AI Raw Data Intelligence Core[/bold] | v2.0")
    console.print("="*60)
    
    root_path = Path(args.root)
    if not root_path.exists():
        console.print(f"[red]Error: Root directory '{args.root}' not found.[/red]")
        sys.exit(1)
        
    for vf_folder in root_path.iterdir():
        if vf_folder.is_dir():
            try:
                process_folder(vf_folder, gsd=args.gsd, use_gemini=args.gemini)
            except Exception as e:
                console.print(f"[red]Failed to process {vf_folder.name}: {str(e)}[/red]")
