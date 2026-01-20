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

# Increase OpenCV pixel limit for large drone orthomosaics
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**32)

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
        # Fallback to root folder if no Ortho subfolder
        ortho_dir = folder
    
    # Support multiple formats
    extensions = ("*.tif", "*.png", "*.jpg", "*.jpeg")
    def find_file(pattern):
        for ext in extensions:
            match = next(ortho_dir.rglob(f"{pattern}{ext}"), None)
            if match: return match
        return None

    op1_path = find_file("*Pre-Pitting*") or find_file("*OP1*") or find_file("*sample_op1*")
    op3_path = find_file("*Post-SW*") or find_file("*Post-Pitting*") or find_file("*OP3*") or find_file("*sample_op3*")
    
    if not op1_path or not op3_path:
        # Check if there are any two images at all as a last resort
        img_files = []
        for ext in extensions:
            img_files.extend(list(ortho_dir.rglob(ext)))
        
        if len(img_files) >= 2:
            op1_path = img_files[0]
            op3_path = img_files[1]
        else:
            console.print(f"[yellow]Warning: Missing image pairs in {folder.name}. Skipping...[/yellow]")
            return

    console.print(f"  [cyan]OP1 (Reference):[/cyan] {op1_path.name}")
    console.print(f"  [cyan]OP3 (Analysis):[/cyan] {op3_path.name}")

    # 2. Pipeline Execution
    import gc
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
        
        # Detection (Done first on OP1)
        task_det = progress.add_task("[green]Identifying Sample Pits...", total=100)
        pits = detect_pits(img1_bytes, gsd_cm_px=gsd)
        progress.update(task_det, completed=100)
        
        # Registration (Align OP3 to OP1)
        task_reg = progress.add_task("[blue]Aligning Temporal Data...", total=100)
        registered_img3 = register_images(img1_bytes, img3_bytes)
        
        # Clear large raw bytes to free ~600MB
        del img1_bytes
        del img3_bytes
        gc.collect()
        
        progress.update(task_reg, completed=100)
        
        # Classification
        task_class = progress.add_task("[magenta]Evaluating Bio-Vitality...", total=len(pits))
        stats = analyze_survival_at_pits(
            registered_img3 if registered_img3 is not None else op3_path.read_bytes(),
            pits,
            gsd_cm_px=gsd,
            use_gemini=use_gemini
        )
        
        # Cleanup final large results
        if registered_img3 is not None:
            del registered_img3
        gc.collect()
        
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
    table = Table(title=f"Advanced Bio-Vitality Audit: {name}", box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Intelligence", style="bold white")
    
    table.add_row("Total Pits Detected", str(stats['total']))
    table.add_row("Alive Saplings", str(stats['total'] - stats['dead']))
    table.add_row("Casualties", str(stats['dead']))
    
    # Advanced GIS Metrics
    rate = stats['rate']
    color = "green" if rate > 80 else "yellow" if rate > 60 else "red"
    table.add_row("Survival Efficiency", f"[{color}]{rate:.2f}%[/{color}]")
    
    # Structural Density (Simulated based on spatial distribution if details present)
    if stats.get('details'):
        coords = np.array([[p['x'], p['y']] for p in stats['details']])
        if len(coords) > 1:
            # Simple spread metric
            spread = np.std(coords, axis=0)
            density = len(coords) / (np.prod(spread) / 1000000 + 1e-6)
            table.add_row("Plantation Density", f"{density:.2f} pits/Mpx")
            table.add_row("Site Uniformity", "High (Industrial Grade)" if np.std(spread) < 500 else "Variable")
    
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
