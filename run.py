
import typer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
import sys
sys.path.append(str(SRC))

app = typer.Typer(help="Word2Vec from Scratch - Run Experiments")

@app.command()
def train(
    run_name: str = typer.Option(..., help="A name for this training run, used for saving the resulting embeddings"),
    embed_dim: int = typer.Option(100, help="Dimensionality of the word embeddings"),
    epochs: int = typer.Option(1, help="Number of training epochs"),
    max_window: int = typer.Option(5, help="Maximum context window size"),
    n_negatives: int = typer.Option(5, help="Number of negative samples to draw per (centre, context) pair")
):
    from training.train import train
    train(run_name, embed_dim, epochs, max_window, n_negatives)


@app.command()
def eval():
    print("Evaluation not implemented yet")
    
    

if __name__ == "__main__":
    app()
    
