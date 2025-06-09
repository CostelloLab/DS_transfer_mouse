"""
This script converts counts to RPKM, row normalizes, and maps gene symbols for
a recount compendium
"""
from typing import Dict, Set

import argparse
import numpy as np
import pandas as pd
from gprofiler import GProfiler


def parse_gene_lengths(file_path: str) -> Dict[str, int]:
    """Parses a tsv file containing genes and their length

    Arguments
    ---------
    file_path - The path to the file mapping genes to lengths

    Returns
    -------
    gene_to_len - A dict mapping ensembl gene ids to their length in base pairs
    """
    gene_to_len = {}
    with open(file_path) as in_file:
        # Throw out header
        in_file.readline()
        for line in in_file:
            line = line.replace('"', '')
            gene, length = line.strip().split('\t')
            try:
                gene_to_len[gene] = int(length)
            except ValueError:
                # Some genes have no length, but will be removed in a later step
                pass
    return gene_to_len


def get_pathway_genes(pathway_file: str) -> Set[str]:
    """
    Read which genes are present in the pathway matrix file

    Arguments
    ---------
    pathway_file: The path to the file storing the pathway matrix as a genes x pathways tsv

    Returns
    -------
    pathway_genes: The set of all genes used in pathways
    """
    with open(pathway_file) as pathway_file:
        pathway_genes = set()

        # Throw out header
        _ = pathway_file.readline()
        for line in pathway_file:
            line = line.strip().split('\t')
            gene = line[0]
            pathway_genes.add(gene)
        return pathway_genes


def calculate_rpkm(counts: np.ndarray, gene_length_arr: np.ndarray) -> np.ndarray:
    """"Given an array of counts, calculate the reads per kilobase million
    based on the steps here:
    https://www.rna-seqblog.com/rpkm-fpkm-and-tpm-clearly-explained/

    Arguments
    ---------
    counts: The array of transcript counts per gene
    gene_length_arr: The array of lengths for each gene in counts

    Returns
    -------
    rpkm: The rpkm normalized expression data
    """
    counts = np.array(counts, dtype=float)

    reads_per_kb = counts / gene_length_arr

    sample_total_counts = np.sum(counts)
    per_million_transcripts = sample_total_counts / 1e6

    rpkm = reads_per_kb / per_million_transcripts

    return rpkm


LINES_IN_FILE = 190112

def calculate_rkpm_normalization(count_file, gene_file, pathway_file, out_file):
    gene_to_len = parse_gene_lengths(gene_file)
    pathway_genes = get_pathway_genes(pathway_file)
    gene_to_use = []

    count_df = pd.read_csv(count_file, sep="\t")
    header_genes_ids = [gene.split('.')[0] for gene in count_df["Geneid"]]
    # print(header_genes_ids)

    header_gene_symbols = []
    gp = GProfiler(return_dataframe=True)
    # Convert Ensembl gene IDs to gene names
    converted_genes = gp.convert(
        organism='mmusculus',  # Specify mouse organism
        query=header_genes_ids,
        target_namespace='MGI'  # Use MGI (Mouse Genome Informatics) symbols for gene names
    )
    
    # Print the converted gene information
    # print(converted_genes)
    gene_id_gene_name_dict = {}
    for index, row in converted_genes.iterrows():
        gene_id_gene_name_dict.update({row['incoming']: row['name']})

    count = 0
    for gene_id in header_genes_ids:
        # If gene name is not found gprofiler will assign the None string as a gene name.
        header_gene_symbols.append(gene_id_gene_name_dict[gene_id])
        if gene_id_gene_name_dict[gene_id]=="None":
            count = count + 1
            # print("None str")
            
    print("Total gene ids not found in gprofiler: ",count, "out of total gene ids", len(header_gene_symbols))
    # print(header_gene_symbols)

    bad_indices = []
    # Keep only the first instance of each gene in the case that multiple
    # Ensembl genes get mapped to one gene symbol
    genes_seen = set()
    for i, gene in enumerate(header_gene_symbols):
        if gene == "None" or gene in genes_seen:
            bad_indices.append(i)
        # Remove genes that aren't in our prior pathways
        elif gene not in pathway_genes:
            bad_indices.append(i)
        else:
            genes_seen.add(gene)

    # Remove genes with unknown lengths
    gene_length_arr = []
    for i, gene in enumerate(header_genes_ids):
        if gene not in gene_to_len.keys():
            bad_indices.append(i)
            gene_length_arr.append(None)
        else:
            gene_length_arr.append(gene_to_len[gene])
            # if gene_to_len[gene]=="N":
            #     print("N found")

    # sort bad_indices and deduplicate
    bad_indices = list(set(bad_indices))
    bad_indices.sort()

    print("Total bad_indices :", len(bad_indices))

    for index in reversed(bad_indices):
            del gene_length_arr[index]
    
    gene_length_arr = np.array(gene_length_arr)

    means = None
    M2 = None

    samples_seen = set()
    # First time through the data, calculate statistics
    # print("bad_indices", bad_indices)
    count_filter_df = count_df.drop(bad_indices).reset_index()
    no_sample_column_label = ["Geneid", "gene_name","chr", "gene_type"]
    count = 0
    for sample in count_filter_df.columns.tolist():
        if sample in no_sample_column_label:
            continue
        try:
            rpkm = calculate_rpkm(count_filter_df[sample], gene_length_arr)
            
            if any(np.isnan(rpkm)):
                continue
    
            # Online variance calculation https://stackoverflow.com/a/15638726/10930590
            if means is None:
                means = rpkm
                M2 = 0
            else:
                delta = rpkm - means
                means = means + delta / (i + 1)
                M2 = M2 + delta * (rpkm - means)
                # print("mean", means)
                # print("M2", M2)
            count = count + 1
        except ValueError as e:
        # Throw out malformed lines caused by issues with downloading data
            print(e)

    per_gene_variances = M2 / count

    # Get tenth percentile variance value
    variance_cutoff = np.percentile(per_gene_variances, 10)
    low_variance_indices = np.where(per_gene_variances < variance_cutoff)[0]

    # Adjust gene length array to match the final genes
    gene_length_arr = np.delete(gene_length_arr, low_variance_indices)

    filtered_variances = np.delete(per_gene_variances, low_variance_indices)
    stds = np.sqrt(filtered_variances)
    filtered_means = np.delete(means, low_variance_indices)

    print(filtered_means.shape)
    print(stds.shape)
    # print(low_variance_indices.shape)
    # print(count_filter_df.shape)
    # print(count_filter_df.head())
    # print(count_filter_df.index.tolist())
    # print(low_variance_indices)
    count_filter_filter_df = count_filter_df.drop(list(low_variance_indices)).reset_index()
    # print(count_filter_filter_df.columns.tolist())
    # print(count_filter_filter_df.index.tolist())
    gene_name_list = count_filter_filter_df["gene_name"]
    out_file_header = 'sample\t'+ "\t".join(gene_name_list)
    # print("Total Headers" , len(gene_name_list) + 1)

    count_filter_filter_rpkm_normalized_df = count_filter_filter_df.copy(deep=True)
    count_filter_filter_rpkm_normalized_df = \
    count_filter_filter_rpkm_normalized_df.drop(no_sample_column_label + ['index', 'level_0'], axis=1)
    # print(count_filter_filter_rpkm_normalized_df.columns.tolist())

    for sample in count_filter_filter_rpkm_normalized_df.columns.tolist():
        try:
            rpkm = calculate_rpkm(count_filter_filter_rpkm_normalized_df[sample], gene_length_arr)
        
            if any(np.isnan(rpkm)):
                continue
    
            normalized_rpkm = (rpkm - filtered_means) / stds
            # Keep only most variable genes
            rpkm_list = normalized_rpkm.tolist()
            count_filter_filter_rpkm_normalized_df[sample] = rpkm_list
        
        except ValueError as e:
            # Throw out malformed lines caused by issues with downloading data
            print(e)    

    count_filter_filter_rpkm_normalized_df["gene_name"] = gene_name_list
    count_filter_filter_rpkm_normalized_df = count_filter_filter_rpkm_normalized_df.rename(columns={"gene_name": "index"}).set_index("index")
    count_filter_filter_rpkm_normalized_df.to_feather(out_file)
    count_filter_filter_rpkm_normalized_df.to_csv(str(out_file).split(".")[0] + ".csv")
    return count_filter_filter_rpkm_normalized_df
