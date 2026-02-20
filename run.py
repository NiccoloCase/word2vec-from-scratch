
import typer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
import sys
sys.path.append(str(SRC))

app = typer.Typer(help="Word2Vec from Scratch - Run Experiments")

@app.command()
def train():
    from training.train import train
    train()


@app.command()
def eval():
    from training.train import train
    train()
    
    

if __name__ == "__main__":
    app()
    
