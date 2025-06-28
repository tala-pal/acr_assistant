import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import time
from math import ceil

def load_dicom_directory(directory_path, progress_callback=None):
    """
    Load DICOM files from a directory and extract metadata
    
    Args:
        directory_path: Path to directory containing DICOM files
        progress_callback: Optional function to report progress
        
    Returns:
        Dictionary with metadata and file paths
    """
    print(f"load_dicom_directory called with progress_callback: {progress_callback is not None}")
    try:
        #if progress_callback:
        #    progress_callback(f"Starting to scan directory: {directory_path}", 0)
            
        # Check if directory exists
        if not os.path.exists(directory_path):
            #if progress_callback:
            #    progress_callback(f"Error: Directory not found: {directory_path}", -1)
                
            return {
                "status": "error",
                "message": f"Directory not found: {directory_path}"
            }
                        
        dicom_files = []
        for file in os.listdir(directory_path):
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(directory_path, file))
                # Update progress more frequently for better feedback
                #if progress_callback and len(dicom_files) % 10 == 0:
                #    progress_callback(f"Found {len(dicom_files)} DICOM files so far...", 40)
                
        if not dicom_files:
            #if progress_callback:
            #    progress_callback(f"Error: No DICOM files found in {directory_path}", -1)
                
            return {
                "status": "error",
                "message": f"No DICOM files found in {directory_path}"
            }
            
        # Sort files by name to ensure they're in the correct order
        dicom_files.sort()
        
        # Read the first file to get metadata
        #if progress_callback:
        #    progress_callback(f"Reading metadata from first file: {os.path.basename(dicom_files[0])}", 60)
            
        try:
            first_dicom = pydicom.dcmread(dicom_files[0])
            #if progress_callback:
            #    progress_callback("Successfully read first DICOM file", 80)
        except Exception as e:
            #if progress_callback:
            #    progress_callback(f"Error reading first DICOM file: {str(e)}", -1)
            return {
                "status": "error",
                "message": f"Error reading first DICOM file: {str(e)}"
            }
        
        # Extract metadata
        #if progress_callback:
        #    progress_callback("Extracting common metadata", 90)
            
        try:
            scan_date = first_dicom.StudyDate if hasattr(first_dicom, 'StudyDate') else "Unknown"
            slice_thickness = first_dicom.SliceThickness if hasattr(first_dicom, 'SliceThickness') else 0
            
            # Calculate number of slices for a 1cm thickness (used in ACR analysis)
            slices_for_1cm = round(10 / slice_thickness) if slice_thickness > 0 else 0
            
            # Get image dimensions
            rows = first_dicom.Rows if hasattr(first_dicom, 'Rows') else 0
            columns = first_dicom.Columns if hasattr(first_dicom, 'Columns') else 0
            image_dimensions = f"{rows}x{columns}"
            
            #if progress_callback:
            #    progress_callback(f"Extracted metadata: {rows}x{columns}, {slice_thickness}mm slices", 95)
        except Exception as e:
            #if progress_callback:
            #    progress_callback(f"Error extracting metadata: {str(e)}", -1)
            return {
                "status": "error",
                "message": f"Error extracting metadata: {str(e)}"
            }
        
        #if progress_callback:
        #    progress_callback("Building results", 99)
            
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
        
        #if progress_callback:
        #    progress_callback("Complete! Ready for analysis.", 100)
            
        return result
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}", -1)
            
        return {
            "status": "error",
            "message": f"Error loading DICOM files: {str(e)}"
        }

'''
def read_dicom_headers(directory_path):
    """
    Load DICOM files from a directory and extract its headers
    
    Args:
        directory_path: Path to directory containing DICOM files
        
    Returns:
        Dictionary with headers and file paths
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
            "tags": {
                "scan_date": scan_date,
                "slice_thickness": slice_thickness,
                "num_slices": len(dicom_files),
                "image_dimensions": image_dimensions,
                "slices_for_1cm": slices_for_1cm
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error when extracting DICOM headers: {str(e)}"
        }
'''

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

'''
def analyze_phantom_slice(file_path, display_figures=True, progress_callback=None):
    """
    Analyze a single ACR phantom slice
    
    Args:
        file_path: Path to the DICOM file
        display_figures: Whether to generate visualization figures
        progress_callback: Optional function to report progress
        
    Returns:
        Dictionary with analysis results
    """
    try:
        if progress_callback:
            progress_callback(f"Loading DICOM file: {os.path.basename(file_path)}", 0)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "status": "error", 
                "message": f"File not found: {file_path}"
            }
        
        # Load the DICOM file
        dicom = pydicom.dcmread(file_path)
        
        if progress_callback:
            progress_callback("Extracting pixel data", 10)
            
        # Get pixel data
        pixel_array = dicom.pixel_array
        
        if progress_callback:
            progress_callback("Analyzing image data", 40)
            
        # Basic image statistics
        min_val = np.min(pixel_array)
        max_val = np.max(pixel_array)
        mean_val = np.mean(pixel_array)
        std_val = np.std(pixel_array)
        
        # More advanced analysis would go here in a real implementation
        # This is a placeholder for actual ACR phantom analysis
        
        # Create figure for visualization if requested
        figure_data = None
        if display_figures:
            if progress_callback:
                progress_callback("Generating visualization", 70)
                
            # Create a simple figure showing the slice
            plt.figure(figsize=(10, 8))
            plt.imshow(pixel_array, cmap='gray')
            plt.colorbar(label='Pixel Value')
            plt.title(f"ACR Phantom Slice: {os.path.basename(file_path)}")
            
            # Convert figure to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            figure_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
        if progress_callback:
            progress_callback("Building results", 90)
            
        # Create result dictionary
        result = {
            "status": "success",
            "message": f"Successfully analyzed {os.path.basename(file_path)}",
            "statistics": {
                "min": float(min_val),
                "max": float(max_val),
                "mean": float(mean_val),
                "std": float(std_val)
            }
        }

        if progress_callback:
            progress_callback("Complete", 100)
            
        return result
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}", -1)
            
        return {
            "status": "error",
            "message": f"Error analyzing DICOM file: {str(e)}"
        }
'''

def analyze_phantom_group(file_paths, display_figures=True, progress_callback=None):
    """
    Analyze a group of ACR phantom slices by summing them
    
    Args:
        file_paths: List of paths to DICOM files to analyze as a group
        display_figures: Whether to generate visualization figures
        progress_callback: Optional function to report progress
        
    Returns:
        Dictionary with analysis results
    """
    try:
        if progress_callback:
            progress_callback(f"Preparing to analyze {len(file_paths)} slices", 5)
            
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
            if progress_callback:
                progress_percentage = 10 + (i / total_files * 40)  # Progress from 10% to 50%
                progress_callback(f"Loading file {i+1}/{total_files}: {os.path.basename(file_path)}", int(progress_percentage))
                
            dicom = pydicom.dcmread(file_path)
            pixel_arrays.append(dicom.pixel_array)
            
        if progress_callback:
            progress_callback("Combining slice data", 60)
            
        # Sum the pixel arrays
        summed_array = np.sum(pixel_arrays, axis=0)
        
        # Basic image statistics
        min_val = np.min(summed_array)
        max_val = np.max(summed_array)
        mean_val = np.mean(summed_array)
        std_val = np.std(summed_array)
        
        # Create figure for visualization if requested
        figure_data = None
        if display_figures:
            if progress_callback:
                progress_callback("Generating visualization", 80)
                
            # Create a simple figure showing the summed slices
            plt.figure(figsize=(10, 8))
            plt.imshow(summed_array, cmap='gray')
            plt.colorbar(label='Summed Pixel Value')
            plt.title(f"Summed ACR Phantom: {len(file_paths)} slices")
            
            # Convert figure to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            figure_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
        if progress_callback:
            progress_callback("Building results", 90)
            
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
            
        if progress_callback:
            progress_callback("Complete", 100)
            
        return result
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}", -1)
            
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
    result = load_dicom_directory(dir_path)
    
    # Print the results
    print(f"Analysis status: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'success':
        print("\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        
        print("\nResults:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # If you want to display the image
        if 'figure' in result:
            print("\nFigure was generated. You can display it using appropriate methods.")
    else:
        print(f"\nError details:\n{result.get('traceback', 'No traceback available')}")