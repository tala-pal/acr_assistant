import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import time
from math import ceil

def load_dicom_directory(directory_path):
    """
    Load DICOM files from a directory and extract metadata
    
    Args:
        directory_path: Path to directory containing DICOM files
        
    Returns:
        Dictionary with metadata and file paths
    """
    print(f"load_dicom_directory called")
    try:            
        # Check if directory exists
        if not os.path.exists(directory_path):                
            return {
                "status": "error",
                "message": f"Directory not found: {directory_path}"
            }
                        
        dicom_files = []
        for file in os.listdir(directory_path):
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(directory_path, file))
                
        if not dicom_files:
            return {
                "status": "error",
                "message": f"No DICOM files found in {directory_path}"
            }
            
        # Sort files by name to ensure they're in the correct order
        dicom_files.sort()
        
        # Read the first file to get metadata
        try:
            first_dicom = pydicom.dcmread(dicom_files[0])
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading first DICOM file: {str(e)}"
            }
        
        # Extract metadata
        try:
            scan_date = first_dicom.StudyDate if hasattr(first_dicom, 'StudyDate') else "Unknown"
            slice_thickness = first_dicom.SliceThickness if hasattr(first_dicom, 'SliceThickness') else 0
            
            # Calculate number of slices for a 1cm thickness (used in ACR analysis)
            slices_for_1cm = round(10 / slice_thickness) if slice_thickness > 0 else 0
            
            # Get image dimensions
            rows = first_dicom.Rows if hasattr(first_dicom, 'Rows') else 0
            columns = first_dicom.Columns if hasattr(first_dicom, 'Columns') else 0
            image_dimensions = f"{rows}x{columns}"
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error extracting metadata: {str(e)}"
            }
        
        # Create result dictionary
        result = {
            "status": "success",
            "message": f"Successfully loaded {len(dicom_files)} DICOM files",
            "metadata": {
                "scan_date": scan_date,
                "slice_thickness": slice_thickness,
                "num_slices": len(dicom_files),
                "image_dimensions": image_dimensions,
                "slices_for_1cm": slices_for_1cm
            },
            "file_paths": dicom_files
        }
            
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading DICOM files: {str(e)}"
        }

def display_slices(directory_path, rows=10, cols=10, figsize=(10, 10)):
    """
    Display multiple DICOM slicds in a grid layout with navigation between pages.
    
    Parameters:
    -----------
    dicom_dir : str
        Path to directory containing DICOM files
    rows : int
        Number of rows in the grid
    cols : int
        Number of columns in the grid
    figsize : tuple
        Figure size (width, height) in inches
    max_images_per_page : int
        Maximum number of images to display per page
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            return {
                "status": "error",
                "message": f"Directory not found: {directory_path}"
            }
                        
        dicom_files = []
        for file in os.listdir(directory_path):
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(directory_path, file))
                
        if not dicom_files:
            return {
                "status": "error",
                "message": f"No DICOM files found in {directory_path}"
            }
            
        # Sort files by name to ensure they're in the correct order
        dicom_files.sort()

        total_images = len(dicom_files)
        print(f"Found {total_images} DICOM files")
        
        # Calculate number of pages needed
        images_per_figure = rows * cols
        n_figures = ceil(total_images / images_per_figure)
            
        print(f"Found {total_images} images. Will create {n_figures} figures with {rows}x{cols} grid.")
            
        # Create figures
        for fig_idx in range(n_figures):
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            axes = axes.flatten()
                
            # Fill the grid with images
            for i in range(images_per_figure):
                img_idx = fig_idx * images_per_figure + i
                    
                if img_idx < total_images:
                    try:
                        img_path = dicom_files[img_idx]
                        ds = pydicom.dcmread(img_path)
                        # Get pixel data and display it
                        pixel_array = ds.pixel_array
                        axes[i].imshow(pixel_array, cmap='gray')
                        axes[i].set_title(f"{img_idx}", fontsize=8)
                    except Exception as e:
                        print(f"Error loading {dicom_files[img_idx]}: {e}")
                        axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                                    ha='center', va='center', color='red')
                
                # Turn off axis for all subplots
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.suptitle(f"Images {fig_idx*images_per_figure+1}-{min((fig_idx+1)*images_per_figure, total_images)}", 
                        fontsize=16, y=1.02)
            plt.show()

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error when displaying slices: {str(e)}"
        }

def analyze_phantom_group(file_paths, display_figures=True):
    """
    Analyze a group of ACR phantom slices by summing them
    
    Args:
        file_paths: List of paths to DICOM files to analyze as a group
        display_figures: Whether to generate visualization figures
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Check if all files exist
        missing_files = [f for f in file_paths if not os.path.exists(f)]
        if missing_files:
            return {
                "status": "error",
                "message": f"Missing files: {', '.join(missing_files)}"
            }
            
        # Load all DICOM files
        pixel_arrays = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            dicom = pydicom.dcmread(file_path)
            pixel_arrays.append(dicom.pixel_array)
            
        # Sum the pixel arrays
        summed_array = np.sum(pixel_arrays, axis=0)
        
        # Basic image statistics
        min_val = np.min(summed_array)
        max_val = np.max(summed_array)
        mean_val = np.mean(summed_array)
        std_val = np.std(summed_array)
        
        # Create figure for visualization if requested
        if display_figures:
            # Create a simple figure showing the summed slices
            plt.figure(figsize=(10, 8))
            plt.imshow(summed_array, cmap='gray')
            plt.colorbar(label='Summed Pixel Value')
            plt.title(f"Summed ACR Phantom: {len(file_paths)} slices")
            plt.savefig('./analyze.png')
            plt.close()
            
        # Create result dictionary
        result = {
            "status": "success",
            "message": f"Successfully analyzed {len(file_paths)} slices",
            "statistics": {
                "min": float(min_val),
                "max": float(max_val),
                "mean": float(mean_val),
                "std": float(std_val)
            },
            "num_slices": len(file_paths)
        }
            
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error analyzing DICOM group: {str(e)}"
        }


def handle_tool_call(tool_name, **kwargs):
    """
    Route tool calls to the appropriate function
    
    Args:
        tool_name: Name of the tool to call
        **kwargs: Arguments for the tool, including optional progress_callback
    
    Returns:
        Result from the tool function
    """
    tools = {
        "load_dicom_directory": load_dicom_directory,
        "display_slices": display_slices,
        "analyze_phantom_group": analyze_phantom_group
    }
    
    if tool_name not in tools:
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}"
        }
    
    return tools[tool_name](**kwargs)

    # Main execution block
if __name__ == "__main__":
    # Path to your test DICOM file
    dir_path = "/Users/Tala/Documents/ACR/012025"  # Replace with your actual file path
    dir_path = "/Users/tamir/workspace/acr_assistant/012025"  # Replace with your actual file path
    
    # Run the function
    result_load = load_dicom_directory(dir_path)
    
    # Print the results
    print(f"Analysis status: {result_load['status']}")
    print(f"Message: {result_load['message']}")
    
    if result_load['status'] == 'success':
        print("\nMetadata:")
        for key, value in result_load['metadata'].items():
            print(f"  {key}: {value}")
        
        print("\nResults:")
        for key, value in result_load.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nError details:\n{result_load.get('traceback', 'No traceback available')}")


    files = ['/Users/tamir/workspace/acr_assistant/012025/BrainMACQ001_PT113.dcm',
             '/Users/tamir/workspace/acr_assistant/012025/BrainMACQ001_PT114.dcm',
             '/Users/tamir/workspace/acr_assistant/012025/BrainMACQ001_PT115.dcm',
             '/Users/tamir/workspace/acr_assistant/012025/BrainMACQ001_PT116.dcm',
             '/Users/tamir/workspace/acr_assistant/012025/BrainMACQ001_PT117.dcm']
    result_analyze = analyze_phantom_group(files)

    if result_analyze['status'] == 'success':
        print("\nMetadata:")
        for key, value in result_analyze['statistics'].items():
            print(f"  {key}: {value}")
        
        print("\nResults:")
        for key, value in result_analyze.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nError details:\n{result_analyze.get('traceback', 'No traceback available')}")
