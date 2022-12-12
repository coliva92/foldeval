# Original author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://gist.github.com/bougui505/5d8bbd714499daa72cc158a0a9a2e1af
# 
# To run this script, type the following in the command line:
# pymol -qc {model.pdb} {reference.pdb} -r pymol_gdt.py

from pymol import cmd
import numpy
from typing import Tuple





@cmd.extend
def superposition_metrics(model: str, 
                          reference: str, 
                          cutoffs: list = [ 1., 2., 4., 8. ],
                          save2file: bool = False,
                          filename: str = 'metrics.txt') -> Tuple[float, float]:
  model += ' and name CA'
  reference += ' and name CA'
  rmsd = cmd.align(model, reference, cycles=0, object='aln')[0]
  mappings = cmd.get_raw_alignment('aln')
  distances = []
  for mapping in mappings:
    atom1 = f"{mapping[0][0]} and id {mapping[0][1]}"
    atom2 = f"{mapping[1][0]} and id {mapping[1][1]}"
    d = cmd.get_distance(atom1, atom2)
    distances.append(d)
  distances = numpy.asarray(distances)
  gdt_scores = []
  for cutoff in cutoffs:
    gdt_scores.append((distances <= cutoff).sum() / float(len(distances)))
  gdt_ts = numpy.mean(gdt_scores)
  out = [ f"GDT_{cutoffs[i]}={gdt_scores[i]:.4f}" for i in range(len(cutoffs)) ]
  print(f" ".join(out))
  print(f"RMSD={rmsd:.4f} GDT={gdt_ts:.4f}")
  if save2file: 
    with open(filename, 'at') as file:
      file.write(f'RMSD   {rmsd:.4f}\n')
      file.write(f'GDT_TS {gdt_ts:.4f}\n')
  return rmsd, gdt_ts



if __name__ == '__main__':
    model, reference = cmd.get_object_list('all')
    gdt_ts, rmsd = superposition_metrics(model, 
                                         reference, 
                                         save2file=True, 
                                         filename='metrics.txt')
