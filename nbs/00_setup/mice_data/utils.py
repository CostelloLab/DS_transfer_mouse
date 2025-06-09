
from functools import lru_cache
from typing import Dict

import biomart


@lru_cache()
def get_ensembl_mappings() -> Dict[str, str]:
    # Set up connection to server
    server = biomart.BiomartServer('http://useast.ensembl.org/biomart')
    mart = server.datasets['mmusculus_gene_ensembl']

    # List the types of data we want
    attributes = ['ensembl_transcript_id', 'mgi_symbol', 'ensembl_gene_id', 'ensembl_peptide_id']

    # Get the mapping between the attributes
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')

    ensembl_to_genesymbol = {}
    # Store the data in a dict
    for line in data.splitlines():
        print("line", line)
        line = line.split('\t')
        transcript_id = line[0]
        gene_symbol = line[1]
        ensembl_gene = line[2]
        ensembl_peptide = line[3]

        ensembl_to_genesymbol[transcript_id] = gene_symbol
        ensembl_to_genesymbol[ensembl_gene] = gene_symbol
        if len(ensembl_peptide) > 0:
            ensembl_to_genesymbol[ensembl_peptide] = gene_symbol

    return ensembl_to_genesymbol
