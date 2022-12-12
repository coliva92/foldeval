from algorithms.generic import PredictionAlgorithm
from IPython.utils import io
import os
import sys
import subprocess
import tqdm.notebook
import enum
from alphafold.notebooks import notebook_utils
import collections
import copy
from concurrent import futures
from urllib import request
import json
import random
from alphafold.data import feature_processing
from alphafold.data import msa_pairing
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data.tools import jackhmmer
from alphafold.common import protein
from alphafold.model import model
from alphafold.model import config
from alphafold.model import data
from alphafold.relax import relax
from alphafold.relax import utils
from matplotlib import gridspec
from IPython import display
from ipywidgets import GridspecLayout
from ipywidgets import Output
import py3Dmol
import matplotlib.pyplot as plt
import numpy as np





class AlphaFold(PredictionAlgorithm):

  def __init__(self):
    super().__init__()
    self.msa_results = None
    self.plddt_bands = [
      (0, 50, '#FF7D45'),
      (50, 70, '#FFDB13'),
      (70, 90, '#65CBF3'),
      (90, 100, '#0053D6')
    ]
    self.to_visualize_pdb = None
    self.pae_outputs = None
    self.plddts = None
    self.unrelaxed_proteins = None
    self.best_model_name = None 
  

  
  def setup(self) -> None: 
    if os.path.isdir('alphafold'): return
    with tqdm.notebook.tqdm(total=200, bar_format=self.tqdm_format) as pbar:
      self._install_dependencies(pbar)
      self._download_repository(pbar)
      self._environment_setup()
    return



  def do_msa(self) -> None:
    global sequences
    model_type_to_use = notebook_utils.ModelType.MONOMER

    ### deepmind/alphafold/notebooks/AlphaFold.ipynb ===========================
    #
    # This is not an officially-supported Google product.
    #
    # This Colab notebook and other information provided is for theoretical 
    # modelling only, caution should be exercised in its use. It is provided 
    # ‘as-is’ without any warranty of any kind, whether expressed or implied. 
    # Information is not intended to be a substitute for professional medical 
    # advice, diagnosis, or treatment, and does not constitute medical or other 
    # professional advice.
    #
    # Copyright 2021 DeepMind Technologies Limited.
    #

    # --- Find the closest source ---
    test_url_pattern = 'https://storage.googleapis.com/alphafold-colab{:s}/latest/uniref90_2021_03.fasta.1'
    ex = futures.ThreadPoolExecutor(3)
    def _af_fetch(source):
      request.urlretrieve(test_url_pattern.format(source))
      return source
    fs = [ex.submit(_af_fetch, source) for source in ['', '-europe', '-asia']]
    source = None
    for f in futures.as_completed(fs):
      source = f.result()
      ex.shutdown()
      break
    JACKHMMER_BINARY_PATH = '/usr/bin/jackhmmer'
    DB_ROOT_PATH = f'https://storage.googleapis.com/alphafold-colab{source}/latest/'
    # The z_value is the number of sequences in a database.
    MSA_DATABASES = [
      {'db_name': 'uniref90',
      'db_path': f'{DB_ROOT_PATH}uniref90_2021_03.fasta',
      'num_streamed_chunks': 59,
      'z_value': 135_301_051},
      {'db_name': 'smallbfd',
      'db_path': f'{DB_ROOT_PATH}bfd-first_non_consensus_sequences.fasta',
      'num_streamed_chunks': 17,
      'z_value': 65_984_053},
      {'db_name': 'mgnify',
      'db_path': f'{DB_ROOT_PATH}mgy_clusters_2019_05.fasta',
      'num_streamed_chunks': 71,
      'z_value': 304_820_129},
    ]
    # Search UniProt and construct the all_seq features only for heteromers, not homomers.
    if model_type_to_use == notebook_utils.ModelType.MULTIMER and len(set(sequences)) > 1:
      MSA_DATABASES.extend([
          # Swiss-Prot and TrEMBL are concatenated together as UniProt.
          {'db_name': 'uniprot',
          'db_path': f'{DB_ROOT_PATH}uniprot_2021_03.fasta',
          'num_streamed_chunks': 98,
          'z_value': 219_174_961 + 565_254},
      ])
    TOTAL_JACKHMMER_CHUNKS = sum(
      [cfg['num_streamed_chunks'] for cfg in MSA_DATABASES])
    MAX_HITS = {
      'uniref90': 10_000,
      'smallbfd': 5_000,
      'mgnify': 501,
      'uniprot': 50_000,
    }
    TQDM_BAR_FORMAT = self.tqdm_format
    def _af_msa_get_msa(fasta_path):
      """Searches for MSA for the given sequence using chunked Jackhmmer search."""
      # Run the search against chunks of genetic databases (since the genetic
      # databases don't fit in Colab disk).
      raw_msa_results = collections.defaultdict(list)
      with tqdm.notebook.tqdm(total=TOTAL_JACKHMMER_CHUNKS, bar_format=TQDM_BAR_FORMAT) as pbar:
        def _af_jackhmmer_chunk_callback(i): pbar.update(n=1)
        for db_config in MSA_DATABASES:
          db_name = db_config['db_name']
          pbar.set_description(f'Searching {db_name}')
          jackhmmer_runner = jackhmmer.Jackhmmer(
              binary_path=JACKHMMER_BINARY_PATH,
              database_path=db_config['db_path'],
              get_tblout=True,
              num_streamed_chunks=db_config['num_streamed_chunks'],
              streaming_callback=_af_jackhmmer_chunk_callback,
              z_value=db_config['z_value'])
          # Group the results by database name.
          raw_msa_results[db_name].extend(jackhmmer_runner.query(fasta_path))
      return raw_msa_results
    features_for_chain = {}
    raw_msa_results_for_sequence = {}
    for sequence_index, sequence in enumerate(sequences, start=1):
      print(f'\nGetting MSA for sequence {sequence_index}')
      fasta_path = f'target_{sequence_index}.fasta'
      with open(fasta_path, 'wt') as f:
        f.write(f'>query\n{sequence}')
      # Don't do redundant work for multiple copies of the same chain in the multimer.
      if sequence not in raw_msa_results_for_sequence:
        raw_msa_results = _af_msa_get_msa(fasta_path=fasta_path)
        raw_msa_results_for_sequence[sequence] = raw_msa_results
      else:
        raw_msa_results = copy.deepcopy(raw_msa_results_for_sequence[sequence])
      # Extract the MSAs from the Stockholm files.
      # NB: deduplication happens later in pipeline.make_msa_features.
      single_chain_msas = []
      uniprot_msa = None
      for db_name, db_results in raw_msa_results.items():
        merged_msa = notebook_utils.merge_chunked_msa(
            results=db_results, max_hits=MAX_HITS.get(db_name))
        if merged_msa.sequences and db_name != 'uniprot':
          single_chain_msas.append(merged_msa)
          msa_size = len(set(merged_msa.sequences))
          print(f'{msa_size} unique sequences found in {db_name} for sequence {sequence_index}')
        elif merged_msa.sequences and db_name == 'uniprot':
          uniprot_msa = merged_msa
      notebook_utils.show_msa_info(single_chain_msas=single_chain_msas, sequence_index=sequence_index)
      # Turn the raw data into model features.
      feature_dict = {}
      feature_dict.update(pipeline.make_sequence_features(
          sequence=sequence, description='query', num_res=len(sequence)))
      feature_dict.update(pipeline.make_msa_features(msas=single_chain_msas))
      # We don't use templates in AlphaFold Colab notebook, add only empty placeholder features.
      feature_dict.update(notebook_utils.empty_placeholder_template_features(
          num_templates=0, num_res=len(sequence)))
      # Construct the all_seq features only for heteromers, not homomers.
      if model_type_to_use == notebook_utils.ModelType.MULTIMER and len(set(sequences)) > 1:
        valid_feats = msa_pairing.MSA_FEATURES + ('msa_species_identifiers',)
        all_seq_features = {
            f'{k}_all_seq': v for k, v in pipeline.make_msa_features([uniprot_msa]).items()
            if k in valid_feats}
        feature_dict.update(all_seq_features)
      features_for_chain[protein.PDB_CHAIN_IDS[sequence_index - 1]] = feature_dict
    # Do further feature post-processing depending on the model type.
    if model_type_to_use == notebook_utils.ModelType.MONOMER:
      self.msa_results = features_for_chain[protein.PDB_CHAIN_IDS[0]]
    elif model_type_to_use == notebook_utils.ModelType.MULTIMER:
      all_chain_features = {}
      for chain_id, chain_features in features_for_chain.items():
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features, chain_id)
      all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
      np_example = feature_processing.pair_and_merge(
          all_chain_features=all_chain_features)
      # Pad MSA to avoid zero-sized extra_msa.
      self.msa_results = pipeline_multimer.pad_msa(np_example, min_num_seq=512)
      # deepmind/alphafold/notebooks/AlphaFold.ipynb ===========================
    return



  def make_prediction(self) -> str: 
    global jobname
    global af_run_relax
    global af_relax_use_gpu
    model_type_to_use = notebook_utils.ModelType.MONOMER
    np_example = self.msa_results
    
    ### deepmind/alphafold/notebooks/AlphaFold.ipynb ===========================
    #
    # This is not an officially-supported Google product.
    #
    # This Colab notebook and other information provided is for theoretical 
    # modelling only, caution should be exercised in its use. It is provided 
    # ‘as-is’ without any warranty of any kind, whether expressed or implied. 
    # Information is not intended to be a substitute for professional medical 
    # advice, diagnosis, or treatment, and does not constitute medical or other 
    # professional advice.
    #
    # Copyright 2021 DeepMind Technologies Limited.
    #

    # --- Run the model ---
    if model_type_to_use == notebook_utils.ModelType.MONOMER:
      model_names = config.MODEL_PRESETS['monomer'] + ('model_2_ptm',)
    elif model_type_to_use == notebook_utils.ModelType.MULTIMER:
      model_names = config.MODEL_PRESETS['multimer']
    output_dir = jobname
    plddts = {}
    ranking_confidences = {}
    pae_outputs = {}
    unrelaxed_proteins = {}
    with tqdm.notebook.tqdm(total=len(model_names) + 1, bar_format=self.tqdm_format) as pbar:
      for model_name in model_names:
        pbar.set_description(f'Running {model_name}')
        cfg = config.model_config(model_name)
        if model_type_to_use == notebook_utils.ModelType.MONOMER:
          cfg.data.eval.num_ensemble = 1
        elif model_type_to_use == notebook_utils.ModelType.MULTIMER:
          cfg.model.num_ensemble_eval = 1
        params = data.get_model_haiku_params(model_name, './alphafold/data')
        model_runner = model.RunModel(cfg, params)
        processed_feature_dict = model_runner.process_features(np_example, random_seed=0)
        prediction = model_runner.predict(processed_feature_dict, random_seed=random.randrange(sys.maxsize))
        mean_plddt = prediction['plddt'].mean()
        if model_type_to_use == notebook_utils.ModelType.MONOMER:
          if 'predicted_aligned_error' in prediction:
            pae_outputs[model_name] = (prediction['predicted_aligned_error'],
                                      prediction['max_predicted_aligned_error'])
          else:
            # Monomer models are sorted by mean pLDDT. Do not put monomer pTM models here as they
            # should never get selected.
            ranking_confidences[model_name] = prediction['ranking_confidence']
            plddts[model_name] = prediction['plddt']
        elif model_type_to_use == notebook_utils.ModelType.MULTIMER:
          # Multimer models are sorted by pTM+ipTM.
          ranking_confidences[model_name] = prediction['ranking_confidence']
          plddts[model_name] = prediction['plddt']
          pae_outputs[model_name] = (prediction['predicted_aligned_error'],
                                    prediction['max_predicted_aligned_error'])
        # Set the b-factors to the per-residue plddt.
        final_atom_mask = prediction['structure_module']['final_atom_mask']
        b_factors = prediction['plddt'][:, None] * final_atom_mask
        unrelaxed_protein = protein.from_prediction(
            processed_feature_dict,
            prediction,
            b_factors=b_factors,
            remove_leading_feature_dimension=(
                model_type_to_use == notebook_utils.ModelType.MONOMER))
        unrelaxed_proteins[model_name] = unrelaxed_protein
        # Delete unused outputs to save memory.
        del model_runner
        del params
        del prediction
        pbar.update(n=1)
      # --- AMBER relax the best model ---
      # Find the best model according to the mean pLDDT.
      best_model_name = max(ranking_confidences.keys(), key=lambda x: ranking_confidences[x])
      if af_run_relax:
        pbar.set_description(f'AMBER relaxation')
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=0,
            tolerance=2.39,
            stiffness=10.0,
            exclude_residues=[],
            max_outer_iterations=3,
            use_gpu=af_relax_use_gpu)
        relaxed_pdb, _, _ = amber_relaxer.process(prot=unrelaxed_proteins[best_model_name])
      else:
        print('Warning: Running without the relaxation stage.')
        relaxed_pdb = protein.to_pdb(unrelaxed_proteins[best_model_name])
      pbar.update(n=1)  # Finished AMBER relax.
    # Construct multiclass b-factors to indicate confidence bands
    # 0=very low, 1=low, 2=confident, 3=very high
    banded_b_factors = []
    for plddt in plddts[best_model_name]:
      for idx, (min_val, max_val, _) in enumerate(self.plddt_bands):
        if plddt >= min_val and plddt <= max_val:
          banded_b_factors.append(idx)
          break
    banded_b_factors = np.array(banded_b_factors)[:, None] * final_atom_mask
    to_visualize_pdb = utils.overwrite_b_factors(relaxed_pdb, banded_b_factors)
    # Write out the prediction
    pred_output_path = os.path.join(output_dir, 'af_prediction.pdb')
    with open(pred_output_path, 'w') as f: f.write(relaxed_pdb)
    # deepmind/alphafold/notebooks/AlphaFold.ipynb =============================
    
    self.plddt_scores = 0.01 * plddts[best_model_name]
    self.pred_output_path = pred_output_path
    self.to_visualize_pdb = to_visualize_pdb
    self.pae_outputs = pae_outputs
    self.plddts = plddts
    self.unrelaxed_proteins = unrelaxed_proteins
    self.best_model_name = best_model_name 

    # se modificó el código original para eleminiar las moléculas OXT que 
    # estaban provocando errores con el programa lddt de OpenStructure
    corrected_output_filename = f'{jobname}/af_prediction_without_oxt_atoms.pdb'
    with open(corrected_output_filename, 'w') as file:
      for line in open(pred_output_path, 'r'):
        if line[:4] == 'ATOM':
          if line.split()[2] != 'OXT': file.write(line)
    return corrected_output_filename



  def display_predicted_model(self) -> None: 
    global jobname
    global dpi
    global show_sidechains
    model_type_to_use = notebook_utils.ModelType.MONOMER
    output_dir = jobname

    ### deepmind/alphafold/notebooks/AlphaFold.ipynb ===========================
    #
    # This is not an officially-supported Google product.
    #
    # This Colab notebook and other information provided is for theoretical 
    # modelling only, caution should be exercised in its use. It is provided 
    # ‘as-is’ without any warranty of any kind, whether expressed or implied. 
    # Information is not intended to be a substitute for professional medical 
    # advice, diagnosis, or treatment, and does not constitute medical or other 
    # professional advice.
    #
    # Copyright 2021 DeepMind Technologies Limited.
    #

    # --- Visualise the prediction & confidence ---
    def _af_plot_plddt_legend():
      """Plots the legend for pLDDT."""
      thresh = ['Very low (pLDDT < 50)',
                'Low (70 > pLDDT > 50)',
                'Confident (90 > pLDDT > 70)',
                'Very high (pLDDT > 90)']

      colors = [x[2] for x in self.plddt_bands]
      plt.figure(figsize=(2, 2))
      for c in colors:
        plt.bar(0, 0, color=c)
      plt.legend(thresh, frameon=False, loc='center', fontsize=20)
      plt.xticks([])
      plt.yticks([])
      ax = plt.gca()
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ax.spines['left'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      plt.title('Model Confidence', fontsize=20, pad=20)
      return plt
    # Show the structure coloured by chain if the multimer model has been used.
    if model_type_to_use == notebook_utils.ModelType.MULTIMER:
      multichain_view = py3Dmol.view(width=800, height=600)
      multichain_view.addModelsAsFrames(self.to_visualize_pdb)
      multichain_style = {'cartoon': {'colorscheme': 'chain'}}
      multichain_view.setStyle({'model': -1}, multichain_style)
      multichain_view.zoomTo()
      multichain_view.show()
    # Color the structure by per-residue pLDDT
    color_map = {i: bands[2] for i, bands in enumerate(self.plddt_bands)}
    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(self.to_visualize_pdb)
    style = {'cartoon': {'colorscheme': {'prop': 'b', 'map': color_map}}}
    if show_sidechains:
      style['stick'] = {}
    view.setStyle({'model': -1}, style)
    view.zoomTo()
    grid = GridspecLayout(1, 2)
    out = Output()
    with out:
      view.show()
    grid[0, 0] = out
    out = Output()
    with out:
      _af_plot_plddt_legend().show()
    grid[0, 1] = out
    display.display(grid)
    # Display pLDDT and predicted aligned error (if output by the model).
    if self.pae_outputs:
      num_plots = 2
    else:
      num_plots = 1
    plt.figure(figsize=[8 * num_plots, 6], dpi=dpi)
    plt.subplot(1, num_plots, 1)
    plt.plot(self.plddt_scores)
    plt.title('Predicted LDDT')
    plt.xlabel('Residue')
    plt.ylabel('pLDDT')
    if num_plots == 2:
      plt.subplot(1, 2, 2)
      pae, max_pae = list(self.pae_outputs.values())[0]
      plt.imshow(pae, vmin=0., vmax=max_pae, cmap='Greens_r')
      plt.colorbar(fraction=0.046, pad=0.04)
      # Display lines at chain boundaries.
      best_unrelaxed_prot = self.unrelaxed_proteins[self.best_model_name]
      total_num_res = best_unrelaxed_prot.residue_index.shape[-1]
      chain_ids = best_unrelaxed_prot.chain_index
      for chain_boundary in np.nonzero(chain_ids[:-1] - chain_ids[1:]):
        if chain_boundary.size:
          plt.plot([0, total_num_res], [chain_boundary, chain_boundary], color='red')
          plt.plot([chain_boundary, chain_boundary], [0, total_num_res], color='red')
      plt.title('Predicted Aligned Error')
      plt.xlabel('Scored residue')
      plt.ylabel('Aligned residue')
    plt.savefig(f"{jobname}/plddt-pae.png", bbox_inches='tight', dpi=dpi)
    # Save the predicted aligned error (if it exists).
    pae_output_path = os.path.join(output_dir, 'predicted_aligned_error.json')
    if self.pae_outputs:
      # Save predicted aligned error in the same format as the AF EMBL DB.
      pae_data = notebook_utils.get_pae_json(pae=pae, max_pae=max_pae.item())
      with open(pae_output_path, 'w') as f:
        f.write(pae_data)
    # deepmind/alphafold/notebooks/AlphaFold.ipynb =============================
    return



  def prepare_results_folder(self) -> str: 
    return super().prepare_results_folder()



  def _install_dependencies(self, pbar) -> None:
    ### deepmind/alphafold/notebooks/AlphaFold.ipynb ===========================
    #
    # This is not an officially-supported Google product.
    #
    # This Colab notebook and other information provided is for theoretical 
    # modelling only, caution should be exercised in its use. It is provided 
    # ‘as-is’ without any warranty of any kind, whether expressed or implied. 
    # Information is not intended to be a substitute for professional medical 
    # advice, diagnosis, or treatment, and does not constitute medical or other 
    # professional advice.
    #
    # Copyright 2021 DeepMind Technologies Limited.
    #
    try:
      with io.capture_output() as captured:
        # Uninstall default Colab version of TF.
        %shell pip uninstall -y tensorflow

        # [HMMER](http://hmmer.org/) es una librería para buscar homólogos 
        # en una base de datos de secuencias, y para hacer alineación de 
        # secuencias usando cadenas ocultas de Markov.
        %shell sudo apt install --quiet --yes hmmer
        pbar.update(6)

        # [Py3DMol](https://github.com/avirshup/py3dmol) es una librería para
        # visualizar moléculas tridimensionales en un iPython Notebook 
        # (_aparentemente está fuera de mantenimiento_).
        %shell pip install py3dmol
        pbar.update(2)

        # Install OpenMM and pdbfixer.
        # [OpenMM](https://openmm.org/) es un toolkit para hacer simulación 
        # molecular.
        # [PDBFixer](https://github.com/openmm/pdbfixer) 
        # es un sanitizador de archivos PDB.
        %shell rm -rf /opt/conda
        %shell wget -q -P /tmp \
          https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
            && rm /tmp/Miniconda3-latest-Linux-x86_64.sh
        pbar.update(9)

        PATH=%env PATH
        %env PATH=/opt/conda/bin:{PATH}
        %shell conda install -qy conda==4.13.0 \
            && conda install -qy -c conda-forge \
              python=3.7 \
              openmm=7.5.1 \
              pdbfixer
        pbar.update(80)

        # Create a ramdisk to store a database chunk to make Jackhmmer run fast.
        # (ver [RAM drive](https://en.wikipedia.org/wiki/RAM_drive)).
        # [Jackhmmer](https://link.springer.com/article/10.1186/1471-2105-11-431)
        # es un algoritmo de búsqueda iterativa de secuencias usando cadenas 
        # ocultas de Markov que además forma parte de OpenMM.
        %shell sudo mkdir -m 777 --parents /tmp/ramdisk
        %shell sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk
        pbar.update(2)

        %shell wget -q -P /content \
          https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
        pbar.update(1)
    except subprocess.CalledProcessError:
      print(captured)
      raise
    # deepmind/alphafold/notebooks/AlphaFold.ipynb =============================
    return
  


  def _download_repository(self, pbar) -> None:
    ### deepmind/alphafold/notebooks/AlphaFold.ipynb ===========================
    #
    # This is not an officially-supported Google product.
    #
    # This Colab notebook and other information provided is for theoretical 
    # modelling only, caution should be exercised in its use. It is provided 
    # ‘as-is’ without any warranty of any kind, whether expressed or implied. 
    # Information is not intended to be a substitute for professional medical 
    # advice, diagnosis, or treatment, and does not constitute medical or other 
    # professional advice.
    #
    # Copyright 2021 DeepMind Technologies Limited.
    #
    GIT_REPO = 'https://github.com/deepmind/alphafold'
    SOURCE_URL = 'https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-03-02.tar'
    PARAMS_DIR = './alphafold/data/params'
    PARAMS_PATH = os.path.join(PARAMS_DIR, os.path.basename(SOURCE_URL))
    try:
      with io.capture_output() as captured:
        %shell rm -rf alphafold
        %shell git clone --branch main {GIT_REPO} alphafold
        pbar.update(8)
        # Install the required versions of all dependencies.
        %shell pip3 install -r ./alphafold/requirements.txt
        # Run setup.py to install only AlphaFold.
        %shell pip3 install --no-dependencies ./alphafold
        pbar.update(10)
        # Consulta [esta página](https://stackoverflow.com/questions/43658870/requirements-txt-vs-setup-py)
        # para conocer la diferencia (teórica) entre ambos archivos.
        # Se desconoce si realmente es necesario ejecutar ambos archivos a
        # pesar de que instalan las mismas dependencias.

        # Apply OpenMM patch.
        # Se desconoce el origen y propósito de este parche
        %shell pushd /opt/conda/lib/python3.7/site-packages/ && \
            patch -p0 < /content/alphafold/docker/openmm.patch && \
            popd

        # Make sure stereo_chemical_props.txt is in all locations where it could be searched for.
        %shell mkdir -p /content/alphafold/alphafold/common
        %shell cp -f /content/stereo_chemical_props.txt /content/alphafold/alphafold/common
        %shell mkdir -p /opt/conda/lib/python3.7/site-packages/alphafold/common/
        %shell cp -f /content/stereo_chemical_props.txt /opt/conda/lib/python3.7/site-packages/alphafold/common/

        # Consulta 
        # [esta página](https://github.com/deepmind/alphafold#model-parameters)
        # para conocer más detalles. La carpeta descargada pesa 3.8GB.
        %shell mkdir --parents "{PARAMS_DIR}"
        %shell wget -O "{PARAMS_PATH}" "{SOURCE_URL}"
        pbar.update(27)

        %shell tar --extract --verbose --file="{PARAMS_PATH}" \
          --directory="{PARAMS_DIR}" --preserve-permissions
        %shell rm "{PARAMS_PATH}"
        pbar.update(55)
    except subprocess.CalledProcessError:
      print(captured)
      raise
    # deepmind/alphafold/notebooks/AlphaFold.ipynb =============================
    return



  def _environment_setup(self) -> None:
    ### deepmind/alphafold/notebooks/AlphaFold.ipynb ===========================
    #
    # This is not an officially-supported Google product.
    #
    # This Colab notebook and other information provided is for theoretical 
    # modelling only, caution should be exercised in its use. It is provided 
    # ‘as-is’ without any warranty of any kind, whether expressed or implied. 
    # Information is not intended to be a substitute for professional medical 
    # advice, diagnosis, or treatment, and does not constitute medical or other 
    # professional advice.
    #
    # Copyright 2021 DeepMind Technologies Limited.
    #
    import jax
    if jax.local_devices()[0].platform == 'tpu':
      raise RuntimeError('Colab TPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
    elif jax.local_devices()[0].platform == 'cpu':
      raise RuntimeError('Colab CPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
    else:
      print(f'Running JAX with {jax.local_devices()[0].device_kind} GPU')
    
    sys.path.append('/opt/conda/lib/python3.7/site-packages')
    sys.path.append('/content/alphafold')

    # Consultar 
    # [esta página](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)
    # para conocer los detalles sobre la configuración de JAX.
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'
    # deepmind/alphafold/notebooks/AlphaFold.ipynb =============================
    return
