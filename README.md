# Proton Agent

Gauge coupling unification, RGE (renormalization group equation running) coefficients, Proton-decay calculations, scans, and MCMC, and plots,
available through the terminal or Open WebUI. Python 3.11 or later is required.

From the downloaded repository, activate your Python environment and install:

```sh
python -m pip install ./main
```

For development, use `python -m pip install -e ./main` instead. Both installation
modes include the required prompts and benchmark examples. With a regular
installation, generated output is saved under the current working directory.
An editable checkout retains its repository output directory. Set
`PROTON_AGENT_WORKDIR` to explicitly choose a working directory in either mode.

For local inference, install and start Ollama and download the model:

```sh
ollama pull gpt-oss:20b
proton_agent --model ollama:gpt-oss:20b
```

Open WebUI is optional. To connect it, start the agent server:

```sh
PROTON_AGENT_SERVER_API_KEY=proton-agent-local \
  proton_agent_server --host 127.0.0.1 --port 8001 \
  --upstream-model ollama:gpt-oss:20b
```

Configure an OpenAI-compatible connection in Open WebUI with the agent server's
`/v1` URL and matching API key, then select `proton-agent`. The server URL must
be reachable from Open WebUI; a container's loopback address refers to the
container itself. Use `--help` for CLI and server provider settings.

The `input/` directory contains four benchmark files and optional
`matrix-elements.txt` and `bounds.txt` templates. Those two optional files are
used only when explicitly requested for the computation. When using installed
benchmark examples outside the checkout, the four packaged benchmarks are
available as fallback inputs; custom files are read from the user's filesystem.

# Citation

If you use proton_agent in your research or reference the LLM-assisted workflow, please cite the accompanying paper: [https://arxiv.org/abs/2610.33593](https://arxiv.org/abs/2610.33593)
```
@article{Saad:2026qmz,
    author = "Saad, Shaikh",
    title = "{A Tool-Grounded AI Agent for Gauge Coupling Unification and Proton Decay}",
    eprint = "2610.33593",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "10",
    year = "2026"
}
```
