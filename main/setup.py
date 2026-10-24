"""Ship the source-tree prompts and examples in wheels and source archives."""

from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
INPUT_NAMES = ("nonsusy-SU5.txt", "nonsusy-SO10.txt", "susy-SU5.txt",
               "susy-SO10.txt", "matrix-elements.txt", "bounds.txt")
RESOURCE_FILES = [Path("system_prompt.txt"), *[Path("prompts") / name for name in (
    "computation_report.txt", "bound_evasion_discussion.txt", "scan_workflow.txt",
    "mcmc_workflow.txt", "rge_agent.txt")],
    *[Path("input") / name for name in INPUT_NAMES]]


def copy_resources(destination):
    for relative in RESOURCE_FILES:
        source = ROOT.parent / relative if relative.parts[0] == "input" else ROOT / relative
        if not source.is_file():
            source = ROOT / "proton_agent_resources" / relative
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)


class BuildWithResources(build_py):
    def run(self):
        super().run()
        copy_resources(Path(self.build_lib) / "proton_agent_resources")


class SourceWithResources(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        copy_resources(Path(base_dir) / "proton_agent_resources")


setup(cmdclass={"build_py": BuildWithResources, "sdist": SourceWithResources})
