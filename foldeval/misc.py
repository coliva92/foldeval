from datetime import datetime
import matplotlib.pyplot as plt





def create_jobname_suffix() -> str:
  now = str(datetime.today())
  return now.replace('-', '').replace(' ', '').replace(':', '').replace('.', '')



def display_lddt_comparison(predicted_scores, 
                            measured_scores, 
                            output_figure_filename: str,
                            dpi: int = 140):
  plt.figure(figsize=(8,5), dpi=dpi)
  plt.plot(predicted_scores, color='#ccc', label='Predicted lDDT')
  plt.plot(measured_scores, label='Measured lDDT')
  plt.xlabel('Residues')
  plt.ylim(0, 1)
  plt.legend()
  plt.savefig(output_figure_filename, bbox_inches='tight')
  plt.show()
