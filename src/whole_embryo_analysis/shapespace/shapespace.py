"""
Main shapespace analysis class.
Follows main workflow:
            1. PCA:
                Runs PCA on the spherical harmonics coefficients from the full dataframe
                to identify the first NUM_PCS principal components. Then, a reverse transform is done
                to map the principal components back to spherical harmonics coefficients. 
            2. digitize_all_pcs shapespace:
                Given the dataframe of principal components as spherical harmonic coefficients,
                all original datapoints are binned within +/- max_sd for each principal component.
                This step creates the shapespace. 
            3. Removes extreme points
                If desired, the top and bottom removal_pct most extreme values are removed.
                This was implemented because of its use in the inspirational paper, but I did not
                notice a signfigant difference between visualized shapespace with our without.  
            4. Stores shapespace if desired
                Saves shapespace as csv.
            5. Visualizates first NUM_PC principal components
                Generates VTK files of principal shapespace, and plots them with pyvista
"""

import os
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import warnings
import matplotlib.pyplot as plt
import polars as pl

warnings.filterwarnings('ignore')

from whole_embryo_analysis.shapespace import figures
LMAX = 16
NUM_PCS = 6
DIR_TAG = 'Shapemodes' # will save a directory in the Sample#Dir as Sample#/DIR_TAG/

@dataclass
class ShapeSpace:
    # === Input Params ===
    # full_path: str
    shcoeffs_df: pd.DataFrame
    save_dir: str
    str_tag: str
    # === Default Params ===
    max_sd: float = 2.0
    store_results: bool = True
    generate_figures: bool = True
    filter_state: bool = False
    lmax: int = LMAX
    num_pcs: int = NUM_PCS
    id_name: Optional[str] = "Nuclear_ID" # column name of the index of the dataframe

    def __post_init__(self):
        """all internal attributes."""
        self.label_index = self.shcoeffs_df.index
        self.pca_df = None
        self.pca = None
        self.df_shapespace = None
        self.pc_scales = {}
        self.meshes = None
        self.removal_pct = 0
        # self.shcoeffs_df = None
        self.save_path = f"{self.save_dir}/{DIR_TAG}_{self.str_tag}"
        if self.generate_figures or self.store_results:
            os.makedirs(self.save_path, exist_ok=True)
        num_points = int(2 * self.max_sd / 0.5) + 1
        self.sigma_steps = np.linspace(-self.max_sd, self.max_sd, num_points).tolist()

    def workflow(self, shcoeffs_df):
        """
        Main driver function of class. Follows main workflow outlined in the class docstring.
        Params:
        ------
            save_dir: str of directory to save results to
            str_tag: str of tag to identify this run
        Returns:
        ------
            None
        """
        self.shcoeffs_df = shcoeffs_df
        print(f"Running PCA with n={self.num_pcs} principal components.")
        self.run_pca()
        print("Digitizing Shapemodes.")
        self.df_shapespace = self.digitize_all_pcs()
        # print(f"Removing the {self.removal_pct}% extremes for mesh generation...")
        # self.df_shapespace = pd.concat([mpids_full, is_extreme], axis=1)
        if self.store_results:
            self.store_shapespace()
        figures.reconstruct_meshes(self)
        if self.generate_figures:
            print("Generating figures.")
            figures.mesh_visualization(self)
            figures.pc_histograms(self)
            figures.variance_explained(self)
            figures.plot_scree(self)

    # ----- Main Shapespace Creation -----
    def run_pca(self):
        """
        Run pca and create spherical harmonic coefficients for each principal component.
        
        Returns:
        ------
            pca_df: pd.DataFrame where there is a column for each principal component and
            each cell has a value for that principal component
        """
        sh_matrix = self.shcoeffs_df.values
        self.pca = PCA(n_components=self.num_pcs, svd_solver='full')
        self.pca.fit(sh_matrix)
        # Project each objects SHE onto the PC axes
        axes = self.pca.transform(sh_matrix)
        # Store as dataframe
        pc_columns = [f"PC{i+1}" for i in range(self.num_pcs)]
        self.pca_df = pd.DataFrame(axes, columns=pc_columns, index=self.label_index)
        print("PCA Complete")
        print(f"Explained variance ratio: ")
        for index, var in enumerate(self.pca.explained_variance_ratio_):
            print(f"PC{index+1}: {100*var:.2f}%")
        print(f"Total explained variance: {100*self.pca.explained_variance_ratio_.sum():.2f}%")
        return self.pca_df
    
    def shapespace_to_shcoeff(self):
        """Converting from PCA space to Spherical Harmonic Space"""
        pc_to_coeffs = []
        pc_names = list(self.pca_df.columns)
        
        for pc_idx, pc_name in enumerate(pc_names):
            # 1. Get the scale (standard deviation)
            scale, _ = self.get_scale(pc_name)
            
            # 2. Create scaled coordinates along this PC
            coords = [s * scale for s in self.sigma_steps]  # e.g., [-2σ, -1.5σ, ..., +2σ]
            
            # 3. Create coordinate matrix (all zeros except for this PC dimension)
            matrix = self.get_coordinates_matrix(coords, pc_idx)
            
            # 4. INVERSE PCA TRANSFORM 
            df_inv = self.invert(matrix)  # ← Converts PCA coords → shcoeffs
            df_inv["shape_mode"] = pc_name
            df_inv["mpId"] = np.arange(1, 1 + len(self.sigma_steps))
            pc_to_coeffs.append(df_inv)
        
        return pd.concat(pc_to_coeffs, ignore_index=True)
    
    def remove_extreme_points(self):
        if not self.removal_pct or self.removal_pct <= 0:
            return self.pca_df
        df = self.pca_df.copy()
        mask = np.zeros(len(df), dtype=bool)
        for col in df.columns:
            ql, qh = np.percentile(df[col].values, [self.removal_pct, 100 - self.removal_pct])
            mask |= (df[col] < ql) | (df[col] > qh)
        extreme_mask = pd.Series(mask, index=self.pca_df.index, name="is_extreme")
        self.pca_df = df.loc[~mask]
        self.shcoeffs_df = self.shcoeffs_df.loc[self.pca_df.index]
        self.label_index = self.pca_df.index
        return extreme_mask

    def get_scale(self, pc, df=None):
        src = self.pca_df if df is None else df
        values = src[pc].values.astype(np.float32)
        values_centered = values - values.mean()
        scale = values_centered.std()
        return scale, values_centered
    
    def digitize_all_pcs(self):
        """
        Driver function to run digitize_pc() for each principal component

        Returns:
        -------
            df with binned shapespace ids added
        """
        df_shapespace = self.pca_df.copy()
        for pc in df_shapespace.columns:
            df_shapespace[f"{pc}_mpId"] = self.digitize_pc(pc, src=df_shapespace)
        return df_shapespace
    
    def digitize_pc(self, pc, src):
        """
        Bin SHE in -/+ sd 

        Params:
            pc: current principal component being digitized
            src: datafame to digitize
        Returns:
            np.array of digitized spherical harmonic coefficients
        """
        scale, values_centered = self.get_scale(pc, df=src)
        if scale == 0:
            normalized_values = values_centered
        else:
            normalized_values = values_centered / scale
        binw = 0.5 * np.diff(self.sigma_steps).mean() if len(self.sigma_steps) > 1 else 1
        bin_edges = np.unique([(b-binw, b+binw) for b in self.sigma_steps])
        bin_edges[0] = -np.inf # each cell's pc value is mapped to a bin
        bin_edges[-1] = np.inf
        return np.digitize(normalized_values, bin_edges)

    
    def get_sigma_range(self, max_sd):
        """
        Return np array of bins for shape space 

        Params:
        ------

        Returns:
        ------
        """
        step = 0.5
        num_points = int(2 * max_sd / step) + 1
        return np.linspace(-max_sd, max_sd, num_points).tolist()

    def store_shapespace(self, filename='shapespace.csv'):
        if self.df_shapespace is None:
            raise ValueError("Shape space not digitized. Please run digitize_all_pcs() first.")
        path_to_save = os.path.join(self.save_path, f'{self.str_tag}_{filename}')
        try:
            self.df_shapespace.to_csv(path_to_save)
            print(f"Successsfuly saved {filename} to {path_to_save}")
        except:
            print(f"Failed to save {filename} to {path_to_save}")
        return
    
    def invert(self, pcs):
        """Matrix has shape NxM, where N is the number of
        samples and M is the number of shape modes."""
        # Inverse PCA here: PCA coords -> shcoeffs
        df = pd.DataFrame(self.pca.inverse_transform(pcs))
        df_print = pl.from_pandas(df)
        # print(df_print.head())
        df.columns = self.shcoeffs_df.columns.tolist()
        return df
    
    def get_coordinates_matrix(self, coords, pc_indx):
        npts = len(coords)
        matrix = np.zeros((npts, self.pca.n_components), dtype=np.float32)
        matrix[:, pc_indx] = coords
        return matrix

    # def plot_scree(self, n_components=50):
    #     sh_matrix = self.shcoeffs_df.values
    #     pca_full = PCA(n_components=n_components, svd_solver='full')
    #     pca_full.fit(sh_matrix)
    #     cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    #     var_explained_chose = self.pca.explained_variance_ratio_.sum()
    #     fig, ax = plt.subplots(figsize=(4, 3))
    #     ax.plot(range(1, len(cumulative_variance) + 1), 
    #             cumulative_variance, 
    #             'o-', 
    #             linewidth=2, 
    #             markersize=8,
    #             color='steelblue')
    #     ax.axvline(x=self.num_shapespace, color='red', linestyle='--', label=f'Selected {self.num_shapespace} components')
    #     ax.axhline(y=var_explained_chose, color='red', linestyle='--', label=f'Variance Explained: {var_explained_chose:.2f}')
    #     ax.legend(loc='best')
    #     ax.set_xlabel('Number of Principal Components')
    #     ax.set_ylabel('Cumulative Variance Explained')
    #     ax.set_title('PCA Scree Plot', fontsize=18)
    #     ax.set_xlim(0, len(cumulative_variance) + 1)
    #     ax.set_ylim(cumulative_variance[0] - 0.05, 1.0)
    #     ax.tick_params(axis='both', which='major')
    #     if self.store_results:
    #         save_vectorized(dir=self.save_path, fig=fig, plot_name='pca_scree_plot')
    #     else:
    #         plt.show()
    