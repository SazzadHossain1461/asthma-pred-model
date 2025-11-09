"""
create_scrna_data.py
Create sample single-cell RNA-seq data for testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import anndata

def create_sample_scrna_data():
    """Create sample single-cell RNA-seq data for testing."""
    print("Creating sample single-cell RNA-seq data...")
    
    # Create output directory
    output_dir = Path("data/scrna_sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters - reduced for faster processing
    n_cells = 500   # Reduced from 1000
    n_genes = 1000  # Reduced from 5000
    n_patients = 8  # Reduced from 10
    
    # Generate random counts matrix
    np.random.seed(42)
    
    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    # Create cell IDs
    cell_ids = [f"Cell_{i:05d}" for i in range(n_cells)]
    
    # Assign patients to cells
    patient_ids = [f"Patient_{i:02d}" for i in range(n_patients)]
    cell_patients = np.random.choice(patient_ids, n_cells)
    
    # Generate counts data (zero-inflated)
    counts_data = np.random.negative_binomial(5, 0.1, (n_cells, n_genes))
    zero_mask = np.random.random((n_cells, n_genes)) > 0.1
    counts_data[zero_mask] = 0
    
    # Create metadata
    metadata = pd.DataFrame({
        'patient_id': cell_patients,
        'batch': np.random.choice(['Batch_A', 'Batch_B'], n_cells),
        'cell_type': np.random.choice(['T_Cell', 'B_Cell', 'Macrophage'], n_cells),
        'condition': np.random.choice(['Asthma', 'Control'], n_cells),
        'n_counts': counts_data.sum(axis=1),
        'n_genes': (counts_data > 0).sum(axis=1)
    }, index=cell_ids)
    
    # Create AnnData object
    adata = anndata.AnnData(X=counts_data.astype(np.float32))
    adata.obs = metadata
    adata.var['gene_ids'] = gene_names
    adata.var['highly_variable'] = np.random.choice([True, False], n_genes, p=[0.1, 0.9])
    
    # Save as AnnData
    adata.write(output_dir / "sample_scrna.h5ad")
    
    print(f"Sample data created in: {output_dir}")
    print(f"Cells: {n_cells}, Genes: {n_genes}, Patients: {n_patients}")
    print(f"Files created:")
    print(f"  - sample_scrna.h5ad (AnnData format)")
    
    return True

if __name__ == "__main__":
    create_sample_scrna_data()