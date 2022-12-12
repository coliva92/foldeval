from algorithms.generic import PredictionAlgorithm
from IPython.utils import io
import os
import sys
import subprocess
import tqdm.notebook
import colabfold as cf
from parsers import parse_a3m
import predict_e2e
import py3Dmol
import matplotlib.pyplot as plt
import numpy as np





class RosettaFold(PredictionAlgorithm):

  def __init__(self):
    super().__init__()



  def setup(self) -> None:
    if os.path.isdir("RoseTTAFold"): return
    with tqdm.notebook.tqdm(total=100, bar_format=self.tqdm_format) as pbar:
      self._install_dependencies(pbar)
      self._environment_setup()
    return



  def do_msa(self) -> None: 
    global rf_msa_method
    global prefix
    global jobname
    global rf_custom_a3m

    ### sokrypton/ColabFold/RoseTTAFold.ipynb ==================================
    #
    # MIT License
    #
    # Copyright (c) 2021 Sergey Ovchinnikov
    #
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    # DEALINGS IN THE SOFTWARE.
    #
    if rf_msa_method == "mmseqs2":
      a3m_lines = cf.run_mmseqs2(sequence, prefix, filter=True)
      with open(f"{jobname}/msa.a3m","w") as a3m: 
        a3m.write(a3m_lines)
    if rf_msa_method == "single_sequence":
      with open(f"{jobname}/msa.a3m","w") as a3m:
        a3m.write(f">{jobname}\n{sequence}\n")
    if rf_msa_method == "custom_a3m":
      # Se modificó esta parte en el código original para que el usuario pueda
      # subir el archivo .a3m mucho antes de llegar a este punto en el código
      msa_dict = rf_custom_a3m
      lines = msa_dict[list(msa_dict.keys())[0]].decode().splitlines()
      a3m_lines = []
      for line in lines:
        line = line.replace("\x00","")
        if len(line) > 0 and not line.startswith('#'):
          a3m_lines.append(line)
      with open(f"{jobname}/msa.a3m","w") as a3m:
        a3m.write("\n".join(a3m_lines))
    self._display_msa_results(f"{jobname}/msa.a3m", 
                              f"{jobname}/msa_coverage.png")
    # sokrypton/ColabFold/RoseTTAFold.ipynb ====================================
    return



  def make_prediction(self) -> str:
    global jobname
    global dpi
    
    ### sokrypton/ColabFold/RoseTTAFold.ipynb ==================================
    #
    # MIT License
    #
    # Copyright (c) 2021 Sergey Ovchinnikov
    #
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    # DEALINGS IN THE SOFTWARE.
    #
    if "rosettafold" not in dir():
      rosettafold = predict_e2e.Predictor(model_dir="weights")
    rosettafold.predict(f"{jobname}/msa.a3m", f"{jobname}/pred")
    self.plddt_scores = _rf_do_scwrl(f"{jobname}/pred.pdb", 
                                     f"{jobname}/rf_prediction.pdb")
    print(f"Predicted LDDT: {self.plddt_scores.mean()}")
    plt.figure(figsize=(8,5), dpi=dpi)
    plt.plot(self.plddt_scores)
    plt.xlabel("Positions")
    plt.ylabel("Predicted lDDT")
    plt.ylim(0, 1)
    plt.savefig(f"{jobname}/plddt.png", bbox_inches='tight')
    plt.show()
    # sokrypton/ColabFold/RoseTTAFold.ipynb ====================================
    return f'{jobname}/rf_prediction.pdb'



  def display_predicted_model(self) -> None: 
    global jobname
    global show_sidechains

    ### sokrypton/ColabFold/RoseTTAFold.ipynb ==================================
    #
    # MIT License
    #
    # Copyright (c) 2021 Sergey Ovchinnikov
    #
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    # DEALINGS IN THE SOFTWARE.
    #
    cf.show_pdb(f"{jobname}/pred.scwrl.pdb", 
                show_sidechains, 
                False, # despliega los átomos del esqueleto
                "lDDT", # coloración de la molécula según el plDDT
                chains=1, 
                vmin=0.5, 
                vmax=0.9).show()
    cf.plot_plddt_legend().show()
    # sokrypton/ColabFold/RoseTTAFold.ipynb ====================================
    return



  def prepare_results_folder(self) -> str: 
    global jobname
    super().prepare_results_folder()

    ### sokrypton/ColabFold/RoseTTAFold.ipynb ==================================
    #
    # MIT License
    #
    # Copyright (c) 2021 Sergey Ovchinnikov
    #
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    # DEALINGS IN THE SOFTWARE.
    #
    settings_path = f"{jobname}/settings.txt"
    if os.path.exists(settings_path): return jobname
    with open(settings_path, "w") as text_file:
      text_file.write(f"method=RoseTTAFold\n")
      text_file.write(f"sequence={sequence}\n")
      text_file.write(f"msa_method={rf_msa_method}\n")
      text_file.write(f"use_templates=False\n")
    # sokrypton/ColabFold/RoseTTAFold.ipynb ====================================
    return jobname



  def _install_dependencies(self, pbar) -> None:
    ### sokrypton/ColabFold/RoseTTAFold.ipynb ==================================
    #
    # MIT License
    #
    # Copyright (c) 2021 Sergey Ovchinnikov
    #
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    # DEALINGS IN THE SOFTWARE.
    #
    import torch
    torch_v = torch.__version__
    try:
      with io.capture_output() as captured:
        # extra functionality
        %shell wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py
        pbar.update(20)

        # download model
        %shell git clone https://github.com/RosettaCommons/RoseTTAFold.git
        %shell wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/RoseTTAFold__network__Refine_module.patch
        %shell patch -u RoseTTAFold/network/Refine_module.py -i RoseTTAFold__network__Refine_module.patch
        pbar.update(20)

        # download model params
        %shell wget -qnc https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz
        %shell tar -xf weights.tar.gz
        %shell rm weights.tar.gz
        pbar.update(20)

        # download scwrl4 (for adding sidechains)
        # http://dunbrack.fccc.edu/SCWRL3.php
        # Thanks Roland Dunbrack!
        # SCWRL4 es un método para predecir conformaciones de cadenas laterales.
        # Lee [este artículo](https://pubmed.ncbi.nlm.nih.gov/19603484/) para 
        # conocer más detalles. 
        %shell wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/scwrl4.zip
        %shell unzip -qqo scwrl4.zip
        pbar.update(20)

        # install libraries
        %shell pip install -q dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
        %shell pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-{torch_v}.html
        %shell pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-{torch_v}.html
        %shell pip install -q torch-geometric
        %shell pip install -q py3Dmol
        pbar.update(20)

        sys.path.append('/content/RoseTTAFold/network')
    except subprocess.CalledProcessError:
      print(captured)
      raise
    # sokrypton/ColabFold/RoseTTAFold.ipynb ====================================
    return



  def _environment_setup(self) -> None:
    global prefix

    ### sokrypton/ColabFold/RoseTTAFold.ipynb ==================================
    #
    # MIT License
    #
    # Copyright (c) 2021 Sergey Ovchinnikov
    #
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    # DEALINGS IN THE SOFTWARE.
    #
    # tmp directory
    os.makedirs('tmp', exist_ok=True)
    prefix = os.path.join('tmp', prefix)
    # sokrypton/ColabFold/RoseTTAFold.ipynb ====================================
    return
  


  def _display_msa_results(self, 
                           msa_filename: str, 
                           output_figure_filename: str) -> None:
    global dpi
    global rf_msa_method

    ### sokrypton/ColabFold/RoseTTAFold.ipynb ==================================
    #
    # MIT License
    #
    # Copyright (c) 2021 Sergey Ovchinnikov
    #
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    # DEALINGS IN THE SOFTWARE.
    #
    msa_all = parse_a3m(msa_filename)
    msa_arr = np.unique(msa_all,axis=0)
    total_msa_size = len(msa_arr)
    if rf_msa_method == "mmseqs2":
      print(f'\n{total_msa_size} total sequences found (after filtering)\n')
    else:
      print(f'\n{total_msa_size} total sequences found\n')
    if total_msa_size > 1:
      plt.figure(figsize=(8,5), dpi=dpi)
      plt.title("Sequence coverage")
      seqid = (msa_all[0] == msa_arr).mean(-1)
      seqid_sort = seqid.argsort()
      non_gaps = (msa_arr != 20).astype(float)
      non_gaps[non_gaps == 0] = np.nan
      plt.imshow(non_gaps[seqid_sort]*seqid[seqid_sort,None],
                 interpolation='nearest', aspect='auto',
                 cmap="rainbow_r", vmin=0, vmax=1, origin='lower',
                 extent=(0, msa_arr.shape[1], 0, msa_arr.shape[0]))
      plt.plot((msa_arr != 20).sum(0), color='black')
      plt.xlim(0, msa_arr.shape[1])
      plt.ylim(0, msa_arr.shape[0])
      plt.colorbar(label="Sequence identity to query",)
      plt.xlabel("Positions")
      plt.ylabel("Sequences")
      plt.savefig(output_figure_filename, bbox_inches='tight')
      plt.show()
    # sokrypton/ColabFold/RoseTTAFold.ipynb ====================================
    return



# Las sigs. funciones deben ser globales debido a un error que surge al 
# invocar subprocess.run()

### sokrypton/ColabFold/RoseTTAFold.ipynb ======================================
#
# MIT License
#
# Copyright (c) 2021 Sergey Ovchinnikov
#
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
#
def _rf_get_bfactor(pdb_filename: str):
  # este código fue modificado de su versión original para corregir un error
  # donde los bfac se copiaban a los residuos incorrectos
  bfac = {}
  for line in open(pdb_filename, "r"):
    if line[:4] == "ATOM":
      res_id = int(line[22:26].strip())
      if res_id in bfac: continue
      bfac[res_id] = float(line[60:66])
  keys = list(bfac.keys())
  keys.sort()
  values = []
  for i in keys: values.append(bfac[i])
  return np.array(values)



def _rf_set_bfactor(pdb_filename: str, bfac) -> None:
  I = open(pdb_filename, "r").readlines()
  O = open(pdb_filename, "w")
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[22:26].strip()) - 1
      O.write(f"{line[:60]}{bfac[seq_id]:6.2f}{line[66:]}")
  O.close()
  return



def _rf_do_scwrl(inputs: str, 
                 outputs: str, 
                 exe: str = "./scwrl4/Scwrl4"):
  subprocess.run([exe, "-i", inputs, "-o", outputs, "-h"], 
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  bfact = _rf_get_bfactor(inputs)
  _rf_set_bfactor(outputs, bfact)
  return bfact
# sokrypton/ColabFold/RoseTTAFold.ipynb ========================================
