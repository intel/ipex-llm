import subprocess
import os

def test_llm_cli_help_fix():
    """
    Tests that the llm-cli script exits gracefully after the fix.
    """
    cli_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/ipex_llm/cli/llm-cli'))
    result = subprocess.run([cli_path, "--help"], capture_output=True, text=True)
    assert "Invalid model_family" not in result.stdout
    assert result.returncode == 0
