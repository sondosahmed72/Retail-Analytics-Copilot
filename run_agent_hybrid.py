"""
Main entrypoint for Retail Analytics Copilot.
Usage: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
"""

import click
import json
import dspy
from rich.console import Console
from rich.progress import Progress
from agent.graph_hybrid import build_hybrid_graph, run_question
import warnings
import logging
import os

# Suppress warnings and logs
warnings.filterwarnings('ignore')
logging.getLogger('litellm').setLevel(logging.CRITICAL)
logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
os.environ['LITELLM_LOG'] = 'ERROR'

console = Console()


def setup_dspy():
    """Initialize DSPy with Ollama/local LLM."""
    # Configure for Ollama with Phi-3.5
    # Use text completion mode to avoid structured output issues
    try:
        lm = dspy.LM(
            model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M",
            api_base="http://localhost:11434",
            max_tokens=1000,
            temperature=0.1,
            cache=False  # Disable caching to avoid errors
        )
        console.print("[green]✓[/green] Using dspy.LM")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Trying alternative configuration...[/yellow]")
        lm = dspy.LM(
            model="ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M",
            api_base="http://localhost:11434",
            max_tokens=1000,
            temperature=0.1
        )
    
    dspy.settings.configure(lm=lm)
    console.print("[green]✓[/green] DSPy configured with Ollama (Phi-3.5)")


@click.command()
@click.option('--batch', required=True, type=click.Path(exists=True), 
              help='Path to input JSONL file with questions')
@click.option('--out', required=True, type=click.Path(), 
              help='Path to output JSONL file')
def main(batch: str, out: str):
    """
    Run Retail Analytics Copilot in batch mode.
    """
    console.print("[bold blue]Retail Analytics Copilot[/bold blue]")
    console.print("=" * 60)
    
    # Setup DSPy
    setup_dspy()
    
    # Build graph
    console.print("[yellow]Building LangGraph...[/yellow]")
    graph = build_hybrid_graph()
    console.print("[green]✓[/green] Graph built successfully")
    
    # Load questions
    console.print(f"[yellow]Loading questions from:[/yellow] {batch}")
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    console.print(f"[green]✓[/green] Loaded {len(questions)} questions")
    
    # Process questions
    results = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing questions...", total=len(questions))
        
        for q in questions:
            question_id = q['id']
            question_text = q['question']
            format_hint = q.get('format_hint', 'str')
            
            console.print(f"\n[bold]Question:[/bold] {question_id}")
            console.print(f"  {question_text}")
            
            # Run question through graph
            try:
                final_state = run_question(graph, question_text, format_hint, question_id)
                
                # Build output
                output = {
                    "id": question_id,
                    "final_answer": final_state.get("final_answer"),
                    "sql": final_state.get("sql_query", ""),
                    "confidence": final_state.get("confidence", 0.0),
                    "explanation": final_state.get("explanation", ""),
                    "citations": final_state.get("citations", [])
                }
                
                results.append(output)
                
                console.print(f"[green]✓[/green] Answer: {output['final_answer']}")
                console.print(f"  Confidence: {output['confidence']:.2f}")
                console.print(f"  Citations: {len(output['citations'])}")
                
                # Debug: show SQL if present
                if output['sql']:
                    console.print(f"  [dim]SQL: {output['sql'][:100]}...[/dim]")
                
                # Print trace for debugging
                if final_state.get("trace"):
                    console.print(f"  Trace: {' → '.join(final_state['trace'][:5])}")
                
            except Exception as e:
                console.print(f"[red]✗[/red] Error: {str(e)}")
                
                # Add error result
                results.append({
                    "id": question_id,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": []
                })
            
            progress.update(task, advance=1)
    
    # Write results
    console.print(f"\n[yellow]Writing results to:[/yellow] {out}")
    with open(out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    console.print(f"[green]✓[/green] Wrote {len(results)} results")
    console.print("\n[bold green]Done![/bold green]")


if __name__ == '__main__':
    main()