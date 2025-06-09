import subprocess
from functools import lru_cache
from pathlib import Path

import nbformat

from pvae import conf


class Notebook:
    """
    This class supports the addition of pytask's tasks that run Jupyter notebooks.
    Notebooks must have a cell with the `inout_defs` tag where the inputs and outputs
    are defined. Optionally, notebook can also have the following cells that are
    relevant in this class:

        1. a cell with the `parameters` tag where the parameters are defined; this is
            important if the inputs or outputs definitions depend on the parameters.
        2. a cell with the `parameters_extra` tag where internal parameters (such as
            a common output directory) are defined; this is important if the inputs or
            outputs definitions depend on the internal parameters.
        3. a cell with the `modules_imports` tag where the modules are imported; this
            is important if the other cells (`parameters`, `parameters_extra`,
            `inout_defs`) depend on some modules (such as pathlib.Path).

    Before reading the `inout_defs` cell, the code given in the `modules_imports`,
    `parameters`, given parameters (as argument to this instance) and `parameters_extra`
    cells is executed (in that order).

    Args:
        input_path: absolute or relative path to the input notebook
        output_path: path to the output notebook relative to the input notebook
        parameters: list of parameters to be passed to the notebook. Each parameter
            must be a string of the form "PARAMETER_NAME PARAMETER_VALUE". For example,
            ["PRED_L 0.0", "KL_RATIO 0.5"].
    """

    def __init__(self, input_path, output_path=None, parameters: list[str] = None):
        self.input_path = Path(input_path).resolve()
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Notebook file not found: {self.input_path}")

        # if the input_path extension is .py, then get the .ipynb file
        if self.input_path.suffix in (".py", ".r", ".R"):
            self.input_path = (
                self.input_path.parent.parent
                / self.input_path.with_suffix(".ipynb").name
            )

        # output_path is always a string, because it's relative to the input notebook
        # (that's how papermill works)
        self.output_path = output_path

        # by default, it is the same as input_path; if parameters are given, it will
        # be output_path
        self.prepared_notebook_path = self.input_path

        self.parameters = parameters
        if self.parameters is None or len(self.parameters) == 0:
            self.parameters_str = ""
        else:
            # prepares the command line parameters to be passed to papermill; each
            # parameter must be a string of the form "PARAMETER_NAME PARAMETER_VALUE"
            self.parameters_str = "-p " + " -p ".join(self.parameters)

        self.notebook_prepared = False

    def prepare(self):
        """
        Runs papermill on the notebook to "prepare" it, i.e., inject the parameters.
        This saves a new version of the notebook in the same input_path or
        output_path (if given).
        """
        if self.parameters_str == "" or self.notebook_prepared:
            self.notebook_prepared = True
            return

        # if there are parameters, then call papermill with --prepare-only to inject
        # parameters
        code_dir = str(conf.common.CODE_DIR)

        output_path = ""
        if self.output_path is not None:
            (self.input_path.parent / self.output_path).parent.mkdir(
                parents=True, exist_ok=True
            )

            output_path = str(self.output_path)

        command = f"""
            bash {code_dir}/scripts/run_nbs.sh {str(self.input_path)} {output_path} --prepare-only {self.parameters_str}
        """.strip()

        subprocess.run(command, shell=True, check=True)

        # set the state of the Notebook as "prepared" and update the path to the
        # prepared notebook.
        self.notebook_prepared = True
        self.prepared_notebook_path = (
            self.input_path.parent / self.output_path
            if self.output_path is not None
            else self.input_path
        )

    def run(self):
        """
        Run the notebook using papermill.
        """
        self.prepare()

        code_dir = str(conf.common.CODE_DIR)
        command = f"""
            bash {code_dir}/scripts/run_nbs.sh {str(self.prepared_notebook_path)} {self.parameters_str}
        """.strip()

        subprocess.run(command, shell=True, check=True)

    def get_kernel(self):
        """
        Get the kernel of the notebook. Only supports Python and R kernels.

        Returns:
            "python" or "r" if the kernel is supported, None otherwise.
        """
        nb = nbformat.read(self.prepared_notebook_path, as_version=nbformat.NO_CONVERT)
        kernel_name = nb.metadata.kernelspec.name
        if "python" in kernel_name:
            return "python"
        elif "r" in kernel_name:
            return "r"

        return None

    def get_injected_parameters_code(self) -> str:
        """
        Return the parameters given as argument of this Notebook object as executable
        Python code.

        Returns:
            Python code that, when executed, defines variables with parameters and
            values given as arguments of this Notebook object.
        """
        if self.parameters_str == "":
            return ""

        def _process_param_value_pair(param_value_pair):
            param, value = param_value_pair.split(" ")

            # if value is not a number, then it must be a string and must be enclosed
            # in quotes
            try:
                float(value)
            except ValueError:
                value = f'"{value}"'

            return f"{param} = {value}"

        return (
            "\n".join(
                [
                    _process_param_value_pair(param_value_pair)
                    for param_value_pair in self.parameters
                ]
            )
            + "\n"
        )

    @lru_cache(maxsize=1)  # noqa: B019
    def get_in_out_defs(self):
        """
        Get the input and output definitions from the notebook. This is a cell in the
        notebook that has the tag "inout_defs" and defines two dictionaries:
        INPUT_FILES and OUTPUT_FILES. The function also reads cells with
        "modules_imports", "parameters" and "parameters_extra" tags from the
        notebook, if any, which are defined by the user. If the notebook is run with
        specific parameters, these are taken into account as papermill will do when
        the notebook is run.

        Returns:
            A tuple of two dictionaries, the first one with the input definitions and
            the second one with the output definitions. Typically, the keys are the
            names of the input/output files and the values are the paths to the files.
        """
        nb = nbformat.read(self.prepared_notebook_path, as_version=nbformat.NO_CONVERT)

        in_out_defs_code = None
        parameters_code = ""
        parameters_extra_code = ""
        modules_code = ""

        # take the code from the cells with the tags "inout_defs", "parameters" and
        # "parameters_extra"
        for cell in nb.cells:
            if cell.metadata.get("tags") == ["inout_defs"] and in_out_defs_code is None:
                in_out_defs_code = cell.source.strip()

            if cell.metadata.get("tags") == ["parameters"]:
                parameters_code += cell.source.strip() + "\n"

            if cell.metadata.get("tags") == ["parameters_extra"]:
                parameters_extra_code += cell.source.strip() + "\n"

            if cell.metadata.get("tags") == ["modules_imports"]:
                modules_code += cell.source.strip() + "\n"

        if in_out_defs_code is None:
            raise ValueError(
                "No input or output definitions found. Add a cell with tag 'inout_defs'"
            )

        # remove lines comments and IPython magic commands from modules_code
        modules_code = "\n".join(
            [
                line
                for line in modules_code.split("\n")
                if not line.startswith(("%", "#"))
            ]
        )

        # combine all the code together; this is needed because the code in
        # "in_out_defs" may depend on the code in "parameters" and "parameters_extra"
        code = "\n".join(
            [
                modules_code,
                parameters_code,
                self.get_injected_parameters_code(),
                parameters_extra_code,
                in_out_defs_code,
            ]
        )

        if self.get_kernel() == "python":
            g = {"conf": conf}
            exec(code, g)
            return g["INPUT_FILES"], g["OUTPUT_FILES"]
        elif self.get_kernel() == "r":
            # TODO: add support to read parameters/injected-parameters as with Python notebooks
            try:
                from rpy2 import robjects

                robjects.r("rm(list=ls())")
                robjects.r(in_out_defs_code)

                g = robjects.r["INPUT_FILES"]
                inputs = {key: Path(g.rx2(key)[0]).resolve() for key in g.names}

                g = robjects.r["OUTPUT_FILES"]
                outputs = {key: Path(g.rx2(key)[0]).resolve() for key in g.names}

                return inputs, outputs
            except ImportError as err:
                raise ImportError("Please install rpy2 to use R notebooks.") from err

    def get_inputs(self) -> dict[str, Path]:
        """
        Get the input definitions from the notebook. It returns the dictionary
        INPUT_FILES from a cell in the notebook with tag "inout_defs". See
        get_in_out_defs() for more details.
        """
        return self.get_in_out_defs()[0]

    def get_outputs(self) -> dict[str, Path]:
        """
        Get the output definitions from the notebook. It returns the dictionary
        OUTPUT_FILES from a cell in the notebook with tag "inout_defs". See
        get_in_out_defs() for more details.
        """
        return self.get_in_out_defs()[1]
