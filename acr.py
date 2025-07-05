import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.ndimage as ndimage
from skimage import measure, filters

def read_dicom_series(directory):
    """
    Read all DICOM files in a directory and return them sorted by instance number.
    """
    dicom_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.dcm'):
            file_path = os.path.join(directory, filename)
            try:
                dicom = pydicom.dcmread(file_path)
                dicom_files.append(dicom)
            except:
                print(f"Could not read {filename} as DICOM")
    
    # Sort by instance number
    if dicom_files and hasattr(dicom_files[0], 'InstanceNumber'):
        dicom_files.sort(key=lambda x: x.InstanceNumber)
    return dicom_files

def calculate_slices_for_1cm_thickness(dicom_series):
    """
    Calculate how many slices need to be summed to achieve approximately 1cm thickness.
    
    Parameters:
    dicom_series (list): List of DICOM datasets
    
    Returns:
    int: Number of slices to sum
    """
    if not dicom_series:
        return None
    
    # Try to get slice thickness from the DICOM tags
    slice_thickness = None
    
    # First, try to get it directly from SliceThickness tag
    if hasattr(dicom_series[0], 'SliceThickness'):
        slice_thickness = dicom_series[0].SliceThickness
    
    # If not available, try to calculate from position information
    elif len(dicom_series) >= 2 and hasattr(dicom_series[0], 'ImagePositionPatient') and hasattr(dicom_series[1], 'ImagePositionPatient'):
        pos1 = dicom_series[0].ImagePositionPatient[2]  # Z-position of first slice
        pos2 = dicom_series[1].ImagePositionPatient[2]  # Z-position of second slice
        slice_thickness = abs(pos2 - pos1)
    
    if slice_thickness:
        # Calculate how many slices for 1cm (10mm)
        slices_for_1cm = round(10.0 / slice_thickness)
        return max(1, slices_for_1cm)  # At least 1 slice
    else:
        # If we can't determine slice thickness, return None
        return None

def draw_circular_roi(image, center, radius):
    """
    Draw a circular ROI on the image and return the ROI mask.
    """
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    cy, cx = center
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = dist <= radius
    return mask

def find_phantom_center(image, threshold=None):
    """
    Find the center of the ACR phantom in an image.
    """
    if threshold is None:
        # Try to find a sensible threshold
        non_zero = image[image > 0]
        if len(non_zero) == 0:
            return None, None
        threshold = np.percentile(non_zero, 25)  # Use 25th percentile as threshold
    
    binary = image > threshold
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_erosion(binary)
    
    # Find the largest connected component (the phantom)
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return None, None
    
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    if len(sizes) == 0:
        return None, None
        
    largest_component = np.argmax(sizes) + 1
    phantom_mask = labeled == largest_component
    
    # Find the center of mass
    cy, cx = ndimage.center_of_mass(phantom_mask)
    return int(cy), int(cx)

def find_hot_cylinder_25mm(image):
    """
    Find the 25mm hot cylinder using geometric properties rather than just max values.
    This function finds the center of the cylinder region, not just the max SUV point.
    
    Parameters:
    image (numpy.ndarray): Image array
    
    Returns:
    tuple: (cy, cx) - center coordinates of the 25mm cylinder
    """
    # Get dimensions
    height, width = image.shape
    
    # Find the phantom center
    center_y, center_x = find_phantom_center(image)
    if center_y is None:
        center_y, center_x = height // 2, width // 2
    
    # Create a mask for the probable phantom region
    phantom_radius = int(height * 0.4)
    phantom_mask = draw_circular_roi(image, (center_y, center_x), phantom_radius)
    
    # Apply mask to image
    masked_image = np.zeros_like(image)
    masked_image[phantom_mask] = image[phantom_mask]
    
    # Find the maximum value and its location
    max_y, max_x = np.unravel_index(np.argmax(masked_image), masked_image.shape)
    max_value = masked_image[max_y, max_x]
    
    # Threshold to find the entire cylinder region - use a range of thresholds
    # Start with a high threshold and gradually lower it until we get a suitable region
    for threshold_factor in [0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
        threshold = max_value * threshold_factor
        cylinder_mask = masked_image > threshold
        
        # Label connected regions
        labeled_regions, num_regions = ndimage.label(cylinder_mask)
        
        # If no regions found, try a lower threshold
        if num_regions == 0:
            continue
        
        # Find the region containing the max value
        max_region_label = labeled_regions[max_y, max_x]
        cylinder_region = labeled_regions == max_region_label
        
        # Check if this region is a suitable size for a cylinder
        region_area = np.sum(cylinder_region)
        expected_area = np.pi * (12.5**2)  # 25mm diameter cylinder area in pixels
        pixel_area = (1.0/image.shape[0])**2  # Approximate pixel area
        
        # If region is too small or too large, try a different threshold
        if region_area < 50 or region_area > 5000:
            continue
        
        # Get properties of the region
        region_props = measure.regionprops(cylinder_region.astype(int))[0]
        
        # Check if the region is reasonably circular
        if region_props.eccentricity > 0.8:  # Too elongated
            continue
            
        # Use the centroid of the region as the cylinder center
        cy, cx = region_props.centroid
        
        # Found a good region, return its center
        return int(cy), int(cx)
    
    # If we couldn't find a good region, fall back to the maximum value position
    print("Warning: Could not find a suitable region for the 25mm cylinder. Using max value position.")
    return max_y, max_x

def display_slice_previews(dicom_series, rows=3, cols=4):
    """
    Display previews of DICOM slices in a grid to help user select which slices to analyze.
    Also calculates how many slices should be summed to achieve 1cm thickness.
    
    Parameters:
    dicom_series (list): List of DICOM datasets
    rows (int): Number of rows in the preview grid
    cols (int): Number of columns in the preview grid
    
    Returns:
    list: List of slice indices selected by the user
    """
    total_slices = len(dicom_series)
    slices_per_page = rows * cols
    total_pages = (total_slices + slices_per_page - 1) // slices_per_page
    
    # Calculate how many slices to sum for 1cm thickness
    slices_for_1cm = calculate_slices_for_1cm_thickness(dicom_series)
    
    if slices_for_1cm:
        print(f"\nINFORMATION: Based on DICOM metadata, you should combine {slices_for_1cm} consecutive slices to achieve 1cm thickness.")
        print(f"When selecting slices, consider choosing groups of {slices_for_1cm} consecutive slices.")
    else:
        print("\nWARNING: Could not determine slice thickness from DICOM metadata.")
        print("Please check your scanner documentation to determine how many slices equal 1cm thickness.")
    
    selected_indices = []
    
    for page in range(total_pages):
        start_idx = page * slices_per_page
        end_idx = min(start_idx + slices_per_page, total_slices)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()
        
        # Display slices in the current page
        for i, ax in enumerate(axes):
            slice_idx = start_idx + i
            if slice_idx < end_idx:
                ax.imshow(dicom_series[slice_idx].pixel_array, cmap='gray')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Preview of Slices {start_idx}-{end_idx-1} (Page {page+1}/{total_pages})", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        # Get user input for slice selection
        print(f"\nViewing slices {start_idx}-{end_idx-1}")
        selection = input("Enter slice numbers to analyze (comma-separated, e.g., '5,8,12') or press Enter to see next page: ")
        
        if selection.strip():
            try:
                indices = [int(idx.strip()) for idx in selection.split(',')]
                valid_indices = [idx for idx in indices if 0 <= idx < total_slices]
                if valid_indices:
                    selected_indices.extend(valid_indices)
                    print(f"Selected slices: {valid_indices}")
                else:
                    print("No valid slice indices entered.")
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers.")
    
    return selected_indices

def calculate_suv(dicom_slice, pixel_data):
    """
    Calculate SUV values from DICOM pixel data.
    
    Parameters:
    dicom_slice: The DICOM dataset containing necessary tags
    pixel_data: The pixel data array after applying rescale slope and intercept
    
    Returns:
    numpy.ndarray: Array of SUV values, or original pixel data if SUV calculation isn't possible
    dict: Information about the SUV calculation
    """
    suv_info = {
        'suv_calculated': False,
        'suv_type': 'bw',  # body weight is the default
        'reason': 'Unknown'
    }
    
    try:
        # First, check if required fields are present
        if not hasattr(dicom_slice, 'PatientWeight'):
            suv_info['reason'] = "Patient weight not found in DICOM"
            return pixel_data, suv_info
        
        # Get patient weight in kg
        weight_kg = float(dicom_slice.PatientWeight)
        suv_info['patient_weight_kg'] = weight_kg
        
        # Check for radiopharmaceutical information
        if not hasattr(dicom_slice, 'RadiopharmaceuticalInformationSequence'):
            suv_info['reason'] = "Radiopharmaceutical information not found"
            return pixel_data, suv_info
        
        rpis = dicom_slice.RadiopharmaceuticalInformationSequence[0]
        
        # Get injected dose
        if hasattr(rpis, 'RadionuclideTotalDose'):
            injected_dose_bq = float(rpis.RadionuclideTotalDose)
        else:
            suv_info['reason'] = "Radionuclide total dose not found"
            return pixel_data, suv_info
        
        suv_info['injected_dose_bq'] = injected_dose_bq
        
        # Get half life of tracer (typically 109.8 minutes for F-18 FDG)
        if hasattr(rpis, 'RadionuclideHalfLife'):
            half_life_seconds = float(rpis.RadionuclideHalfLife)
        else:
            # Default for F-18 if not specified
            half_life_seconds = 6588.0  # 109.8 minutes
        
        suv_info['half_life_seconds'] = half_life_seconds
        
        # Get injection time
        if hasattr(rpis, 'RadiopharmaceuticalStartTime'):
            injection_time = rpis.RadiopharmaceuticalStartTime
        else:
            suv_info['reason'] = "Radiopharmaceutical start time not found"
            return pixel_data, suv_info
        
        # Convert injection time to seconds since midnight
        injection_hours = int(injection_time[0:2])
        injection_minutes = int(injection_time[2:4])
        injection_seconds = int(injection_time[4:6]) if len(injection_time) > 5 else 0
        injection_seconds_total = injection_hours * 3600 + injection_minutes * 60 + injection_seconds
        
        # Get acquisition time
        if hasattr(dicom_slice, 'SeriesTime'):
            acquisition_time = dicom_slice.SeriesTime
        elif hasattr(dicom_slice, 'AcquisitionTime'):
            acquisition_time = dicom_slice.AcquisitionTime
        else:
            suv_info['reason'] = "Acquisition time not found"
            return pixel_data, suv_info
        
        # Convert acquisition time to seconds since midnight
        acquisition_hours = int(acquisition_time[0:2])
        acquisition_minutes = int(acquisition_time[2:4])
        acquisition_seconds = int(acquisition_time[4:6]) if len(acquisition_time) > 5 else 0
        acquisition_seconds_total = acquisition_hours * 3600 + acquisition_minutes * 60 + acquisition_seconds
        
        # Handle case where acquisition is on the next day
        if acquisition_seconds_total < injection_seconds_total:
            acquisition_seconds_total += 24 * 3600  # Add 24 hours
        
        # Calculate decay time and factor
        decay_time_seconds = acquisition_seconds_total - injection_seconds_total
        decay_factor = 2 ** (-decay_time_seconds / half_life_seconds)
        decayed_dose_bq = injected_dose_bq * decay_factor
        
        suv_info['decay_time_seconds'] = decay_time_seconds
        suv_info['decay_factor'] = decay_factor
        suv_info['decayed_dose_bq'] = decayed_dose_bq
        
        # Calculate SUV factor based on body weight
        # SUV = Activity Concentration (Bq/mL) / (Injected Dose (Bq) * Decay Factor / Patient Weight (g))
        suv_factor = weight_kg * 1000 / decayed_dose_bq  # Convert kg to g
        
        # Apply the SUV conversion
        suv_values = pixel_data * suv_factor
        
        suv_info['suv_factor'] = suv_factor
        suv_info['suv_calculated'] = True
        
        print("SUV calculation successful:")
        print(f"  Patient Weight: {weight_kg:.1f} kg")
        print(f"  Injected Dose: {injected_dose_bq/1e6:.2f} MBq")
        print(f"  Decay Time: {decay_time_seconds/60:.1f} minutes")
        print(f"  Decay Factor: {decay_factor:.4f}")
        print(f"  Decay-Corrected Dose: {decayed_dose_bq/1e6:.2f} MBq")
        print(f"  SUV Factor: {suv_factor:.6f}")
        
        return suv_values, suv_info
        
    except Exception as e:
        suv_info['reason'] = f"Error calculating SUV: {str(e)}"
        print(f"Could not calculate SUV: {e}")
        print("Using raw pixel values instead.")
        return pixel_data, suv_info

def find_all_hot_cylinders(image, center_y, center_x, pixel_spacing, display_debug=False):
    """
    Find all hot cylinders in the image using correct positioning:
    - All cylinders are at the same distance from center
    - 25mm cylinder is found as brightest region (not just point)
    - 16mm cylinder is at 45° from 25mm
    - 12mm cylinder is at 90° from 25mm
    - 8mm cylinder is at 22.5° from 12mm (112.5° from 25mm)
    """
    cylinders = {}
    height, width = image.shape
    
    # Create a mask for the phantom region
    phantom_radius = int(min(height, width) * 0.45)
    phantom_mask = draw_circular_roi(image, (center_y, center_x), phantom_radius)
    
    # Apply mask to image
    masked_image = np.zeros_like(image)
    masked_image[phantom_mask] = image[phantom_mask]
    
    # Find the 25mm cylinder as the region with highest average intensity
    # Create a blurred version to help identify regions
    from scipy import ndimage
    blurred_image = ndimage.gaussian_filter(masked_image, sigma=2)
    
    # Define ROI size for 25mm cylinder
    roi_radius_25mm = int(25 / (2 * pixel_spacing))
    
    # Search for the region with highest average value
    best_avg = 0
    best_pos = None
    
    # Create grid of potential center points
    y_indices, x_indices = np.where(phantom_mask)
    
    # Sample points for efficiency
    sample_step = 3  # Check every 5th point
    for i in range(0, len(y_indices), sample_step):
        y, x = y_indices[i], x_indices[i]
        # Skip points too close to edge
        if y < roi_radius_25mm or y >= height - roi_radius_25mm or \
           x < roi_radius_25mm or x >= width - roi_radius_25mm:
            continue
        
        # Create ROI mask
        roi_mask = draw_circular_roi(image, (y, x), roi_radius_25mm)
        # Calculate average intensity
        avg_intensity = np.mean(image[roi_mask])
        
        # Track the best region
        if avg_intensity > best_avg:
            best_avg = avg_intensity
            best_pos = (y, x)
    
    # If we found a valid region
    if best_pos is not None:
        y_25mm, x_25mm = best_pos
        cylinders['cylinder_25mm'] = ((y_25mm, x_25mm), "25mm Cylinder")
        
        # Calculate angle and distance from center to 25mm cylinder
        dx_25mm = x_25mm - center_x
        dy_25mm = y_25mm - center_y
        distance_from_center = np.sqrt(dx_25mm**2 + dy_25mm**2)
        angle_25mm = np.arctan2(dy_25mm, dx_25mm)
        
        # Now find other cylinders based on this 25mm position
        # All other cylinders will be at the same distance from center
        
        # 16mm cylinder at 45° from 25mm
        angle_16mm = angle_25mm + (np.pi / 4)  # 45 degrees
        x_16mm = center_x + distance_from_center * np.cos(angle_16mm)
        y_16mm = center_y + distance_from_center * np.sin(angle_16mm)
        
        # Define search region for 16mm
        search_radius = int(25 / pixel_spacing)  # Larger than 16mm
        roi_radius_16mm = int(16 / (2 * pixel_spacing))
        
        # Find the best position within search region
        search_mask = draw_circular_roi(np.zeros_like(image), 
                                      (int(y_16mm), int(x_16mm)), 
                                      search_radius)
        best_avg = 0
        best_pos = None
        
        for sy, sx in zip(*np.where(search_mask)):
            # Skip points too close to edge
            if sy < roi_radius_16mm or sy >= height - roi_radius_16mm or \
               sx < roi_radius_16mm or sx >= width - roi_radius_16mm:
                continue
                
            # Create ROI mask
            roi_mask = draw_circular_roi(image, (sy, sx), roi_radius_16mm)
            # Calculate average intensity
            avg_intensity = np.mean(image[roi_mask])
            
            # Track the best region
            if avg_intensity > best_avg:
                best_avg = avg_intensity
                best_pos = (sy, sx)
        
        if best_pos is not None:
            y_16mm, x_16mm = best_pos
            cylinders['cylinder_16mm'] = ((y_16mm, x_16mm), "16mm Cylinder")
        
        # 12mm cylinder at 90° from 25mm
        angle_12mm = angle_25mm + (np.pi / 2)  # 90 degrees
        x_12mm = center_x + distance_from_center * np.cos(angle_12mm)
        y_12mm = center_y + distance_from_center * np.sin(angle_12mm)
        
        # Define search region for 12mm
        search_radius = int(20 / pixel_spacing)
        roi_radius_12mm = int(12 / (2 * pixel_spacing))
        
        # Find the best position within search region
        search_mask = draw_circular_roi(np.zeros_like(image), 
                                      (int(y_12mm), int(x_12mm)), 
                                      search_radius)
        best_avg = 0
        best_pos = None
        
        for sy, sx in zip(*np.where(search_mask)):
            # Skip points too close to edge
            if sy < roi_radius_12mm or sy >= height - roi_radius_12mm or \
               sx < roi_radius_12mm or sx >= width - roi_radius_12mm:
                continue
                
            # Create ROI mask
            roi_mask = draw_circular_roi(image, (sy, sx), roi_radius_12mm)
            # Calculate average intensity
            avg_intensity = np.mean(image[roi_mask])
            
            # Track the best region
            if avg_intensity > best_avg:
                best_avg = avg_intensity
                best_pos = (sy, sx)
        
        if best_pos is not None:
            y_12mm, x_12mm = best_pos
            cylinders['cylinder_12mm'] = ((y_12mm, x_12mm), "12mm Cylinder")
        
        # 8mm cylinder at 22.5° from 12mm (112.5° from 25mm)
        angle_8mm = angle_25mm + (np.pi / 2) + (np.pi / 8)  # 90° + 22.5° = 112.5°
        x_8mm = center_x + distance_from_center * np.cos(angle_8mm)
        y_8mm = center_y + distance_from_center * np.sin(angle_8mm)
        
        # Define search region for 8mm
        search_radius = int(15 / pixel_spacing)
        roi_radius_8mm = int(8 / (2 * pixel_spacing))
        
        # Find the best position within search region
        search_mask = draw_circular_roi(np.zeros_like(image), 
                                      (int(y_8mm), int(x_8mm)), 
                                      search_radius)
        best_avg = 0
        best_pos = None
        
        for sy, sx in zip(*np.where(search_mask)):
            # Skip points too close to edge
            if sy < roi_radius_8mm or sy >= height - roi_radius_8mm or \
               sx < roi_radius_8mm or sx >= width - roi_radius_8mm:
                continue
                
            # Create ROI mask
            roi_mask = draw_circular_roi(image, (sy, sx), roi_radius_8mm)
            # Calculate average intensity
            avg_intensity = np.mean(image[roi_mask])
            
            # Track the best region
            if avg_intensity > best_avg:
                best_avg = avg_intensity
                best_pos = (sy, sx)
        
        if best_pos is not None:
            y_8mm, x_8mm = best_pos
            cylinders['cylinder_8mm'] = ((y_8mm, x_8mm), "8mm Cylinder")
    
    # Create visualization for debugging
    if display_debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 12))
        
        # Plot the original image
        plt.subplot(2, 2, 1)
        plt.imshow(masked_image, cmap='gray')
        plt.title('Original Image')
        plt.colorbar()
        
        # Plot the detected cylinders
        plt.subplot(2, 2, 2)
        plt.imshow(masked_image, cmap='gray')
        plt.title('Detected Cylinders')
        
        colors = {'cylinder_25mm': 'red', 'cylinder_16mm': 'blue', 
                 'cylinder_12mm': 'orange', 'cylinder_8mm': 'purple'}
        
        # Draw phantom center
        plt.scatter(center_x, center_y, c='white', s=100, marker='+', label='Center')
        
        # Draw all detected cylinders
        for cylinder_id, (pos, desc) in cylinders.items():
            cy, cx = pos
            plt.scatter(cx, cy, c=colors.get(cylinder_id, 'white'), s=80, marker='o', label=desc)
            
            # Draw circle representing ROI size
            size_mm = int(cylinder_id.split('_')[1].split('mm')[0])
            roi_radius = int(size_mm / (2 * pixel_spacing))
            circle = plt.Circle((cx, cy), roi_radius, fill=False, 
                              edgecolor=colors.get(cylinder_id, 'white'), linewidth=2)
            plt.gca().add_patch(circle)
        
        # Draw circle showing distance from center
        if 'cylinder_25mm' in cylinders:
            cy_25mm, cx_25mm = cylinders['cylinder_25mm'][0]
            dist = np.sqrt((cx_25mm - center_x)**2 + (cy_25mm - center_y)**2)
            circle = plt.Circle((center_x, center_y), dist, fill=False, 
                              edgecolor='white', linestyle='--', linewidth=1)
            plt.gca().add_patch(circle)
        
        # Draw expected positions
        plt.subplot(2, 2, 3)
        plt.imshow(masked_image, cmap='gray')
        plt.title('Angular Relationships')
        
        # Draw phantom center
        plt.scatter(center_x, center_y, c='white', s=100, marker='+', label='Center')
        
        if 'cylinder_25mm' in cylinders:
            cy_25mm, cx_25mm = cylinders['cylinder_25mm'][0]
            angle_25mm = np.arctan2(cy_25mm - center_y, cx_25mm - center_x)
            dist = np.sqrt((cx_25mm - center_x)**2 + (cy_25mm - center_y)**2)
            
            # Draw 25mm position
            plt.scatter(cx_25mm, cy_25mm, c='red', s=80, marker='o', label='25mm')
            
            # Draw circle showing distance from center
            circle = plt.Circle((center_x, center_y), dist, fill=False, 
                             edgecolor='white', linestyle='--', linewidth=1)
            plt.gca().add_patch(circle)
            
            # Draw lines showing angles
            plt.plot([center_x, cx_25mm], [center_y, cy_25mm], 'r-', alpha=0.7, linewidth=2)
            
            # 16mm at 45°
            angle_16mm = angle_25mm + (np.pi / 4)
            x_16mm_exp = center_x + dist * np.cos(angle_16mm)
            y_16mm_exp = center_y + dist * np.sin(angle_16mm)
            plt.plot([center_x, x_16mm_exp], [center_y, y_16mm_exp], 'b-', alpha=0.7, linewidth=2)
            plt.text(x_16mm_exp+10, y_16mm_exp, "45°", color='blue', fontsize=12)
            
            # 12mm at 90°
            angle_12mm = angle_25mm + (np.pi / 2)
            x_12mm_exp = center_x + dist * np.cos(angle_12mm)
            y_12mm_exp = center_y + dist * np.sin(angle_12mm)
            plt.plot([center_x, x_12mm_exp], [center_y, y_12mm_exp], 'y-', alpha=0.7, linewidth=2)
            plt.text(x_12mm_exp+10, y_12mm_exp, "90°", color='orange', fontsize=12)
            
            # 8mm at 112.5° (22.5° from 12mm)
            angle_8mm = angle_25mm + (np.pi / 2) + (np.pi / 8)
            x_8mm_exp = center_x + dist * np.cos(angle_8mm)
            y_8mm_exp = center_y + dist * np.sin(angle_8mm)
            plt.plot([center_x, x_8mm_exp], [center_y, y_8mm_exp], 'm-', alpha=0.7, linewidth=2)
            plt.text(x_8mm_exp+10, y_8mm_exp, "112.5°", color='purple', fontsize=12)
            
            # Draw arc to show 22.5° from 12mm to 8mm
            if 'cylinder_12mm' in cylinders:
                cy_12mm, cx_12mm = cylinders['cylinder_12mm'][0]
                r = dist * 0.3  # Radius for the arc
                
                # Draw arc from 12mm to 8mm
                theta1 = np.degrees(angle_12mm)
                theta2 = np.degrees(angle_8mm)
                arc = plt.matplotlib.patches.Arc((center_x, center_y), 
                                              r*2, r*2, 
                                              theta1=theta1, theta2=theta2,
                                              linewidth=2, color='purple')
                plt.gca().add_patch(arc)
                
                # Label the 22.5° angle
                mid_angle = (angle_12mm + angle_8mm) / 2
                mid_x = center_x + (r * 1.2) * np.cos(mid_angle)
                mid_y = center_y + (r * 1.2) * np.sin(mid_angle)
                plt.text(mid_x, mid_y, "22.5°", color='purple', fontsize=10)
        
        # Plot averages of regions
        plt.subplot(2, 2, 4)
        plt.imshow(blurred_image, cmap='gray')
        plt.title('Blurred Image for Region Detection')
        plt.colorbar()
        
        # Only add legend to subplots that have labeled elements
        plt.subplot(2, 2, 2)  # Detected cylinders subplot
        plt.legend(loc='upper right')

        plt.subplot(2, 2, 3)  # Angular relationships subplot
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    return cylinders

def analyze_acr_phantom(dicom_series, display_figures=True, **kwargs):
    """
    Analyze an ACR phantom DICOM series with mixed evaluation criteria.
    """
    results = {}
    
    # Extract additional parameters from kwargs
    for key, value in kwargs.items():
        results[key] = value
    
    # Use the first slice in the provided series
    slice_data = dicom_series[0]
    
    # Extract pixel data
    pixel_data = slice_data.pixel_array.astype(float)
    
    # Apply rescale slope and intercept if available
    rescale_slope = getattr(slice_data, 'RescaleSlope', 1.0)
    rescale_intercept = getattr(slice_data, 'RescaleIntercept', 0.0)
    pixel_data = pixel_data * rescale_slope + rescale_intercept
    
    # Calculate SUV values if possible
    suv_data, suv_info = calculate_suv(slice_data, pixel_data)
    
    # Store SUV calculation info
    results['suv_calculated'] = suv_info['suv_calculated']
    if suv_info['suv_calculated']:
        results['suv_info'] = suv_info
        
    # Find phantom center
    center_y, center_x = find_phantom_center(suv_data)
    if center_y is None:
        raise ValueError("Could not find phantom center")
    
    # 1. Draw background ROI in center (6-7 cm diameter)
    pixel_spacing = slice_data.PixelSpacing[0]  # Get pixel spacing in mm.  Assuming square pixels
    bg_diameter_mm = 65  # 6.5 cm diameter (average of 6-7 cm)
    bg_radius_pixels = int(bg_diameter_mm / (2 * pixel_spacing))
    background_mask = draw_circular_roi(suv_data, (center_y, center_x), bg_radius_pixels)
    background_mean = np.mean(suv_data[background_mask])
    background_max = np.max(suv_data[background_mask])
    results['background_mean'] = background_mean
    results['background_max'] = background_max
    results['bg_diameter_mm'] = bg_diameter_mm
    
    # 2. Find all hot cylinders using the new function
    # Enable debug visualization if requested
   # In analyze_acr_phantom function, replace:
    known_cylinders = find_all_hot_cylinders(suv_data, center_y, center_x, pixel_spacing, 
                                       display_debug=display_figures)
    
   
    # Process all known cylinders
    cylinder_diameters = {
        'cylinder_25mm': 25,
        'cylinder_16mm': 25,
        'cylinder_12mm': 25,
        'cylinder_8mm': 25
    }
    
    cylinder_colors = {
        'cylinder_25mm': 'red',
        'cylinder_16mm': 'blue',
        'cylinder_12mm': 'orange',
        'cylinder_8mm': 'purple'
    }
    
    # Measure all cylinders
    cylinder_measurements = {}
    for cylinder_id, (pos, desc) in known_cylinders.items():
        cy, cx = pos
        diameter_mm = cylinder_diameters.get(cylinder_id, 25)
        radius_pixels = int(diameter_mm / (2 * pixel_spacing))
        
        # Create ROI mask
        cylinder_mask = draw_circular_roi(suv_data, (cy, cx), radius_pixels)
        cylinder_mean = np.mean(suv_data[cylinder_mask])
        cylinder_max = np.max(suv_data[cylinder_mask])
        
        # Store measurements
        cylinder_measurements[cylinder_id] = {
            'position': pos,
            'description': desc,
            'diameter_mm': diameter_mm,
            'radius_pixels': radius_pixels,
            'mean': cylinder_mean,
            'max': cylinder_max
        }
    
    # Store the measurements in results
    for cylinder_id, data in cylinder_measurements.items():
        results[f"{cylinder_id}_mean"] = data['mean']
        results[f"{cylinder_id}_max"] = data['max']
        results[f"{cylinder_id}_diameter"] = data['diameter_mm']
    
    # Calculate 16mm/25mm ratio if both cylinders were found
    if 'cylinder_25mm' in cylinder_measurements and 'cylinder_16mm' in cylinder_measurements:
        ratio_16_25_mean = cylinder_measurements['cylinder_16mm']['mean'] / cylinder_measurements['cylinder_25mm']['mean']
        ratio_16_25_max = cylinder_measurements['cylinder_16mm']['max'] / cylinder_measurements['cylinder_25mm']['max']
        
        results['ratio_16_25_mean'] = ratio_16_25_mean
        results['ratio_16_25_max'] = ratio_16_25_max
        
        # For backward compatibility
        results['ratio_16_25'] = ratio_16_25_max  # Using max values
    
    # For backward compatibility
    results['background_suv'] = results['background_mean']  # Using mean for background
    if 'cylinder_25mm_max' in results:
        results['cylinder_25mm_suv'] = results['cylinder_25mm_max']  # Using max for cylinders
    if 'cylinder_16mm_max' in results:
        results['cylinder_16mm_suv'] = results['cylinder_16mm_max']  # Using max for cylinders
    
    # PASS/FAIL evaluation - MIXED CRITERIA
    # Background uses mean values
    results['background_pass'] = 0.90 <= background_mean <= 1.10
    
    # 25mm cylinder uses max values
    if 'cylinder_25mm_max' in results:
        results['cylinder_25mm_pass'] = 1.87 <= results['cylinder_25mm_max'] <= 2.91
    
    # 16mm/25mm ratio
    if 'ratio_16_25_max' in results:
        results['ratio_16_25_pass'] = results['ratio_16_25_max'] > 0.7
    
    

    # Only create and display figures if requested
    if display_figures:
        # Create a single figure with multiple subplots
        fig = plt.figure(figsize=(18, 10))
        
        # Get the scan date from the DICOM
        scan_date = "Unknown Date"
        if hasattr(slice_data, 'StudyDate'):
            date_str = slice_data.StudyDate
            if len(date_str) == 8:  # YYYYMMDD format
                scan_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        elif hasattr(slice_data, 'AcquisitionDate'):
            date_str = slice_data.AcquisitionDate
            if len(date_str) == 8:  # YYYYMMDD format
                scan_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Use appropriate label based on whether SUV was calculated
        value_label = 'SUV' if suv_info['suv_calculated'] else 'Pixel Value'
        
        # 1. Original image with ROIs
        ax1 = plt.subplot(2, 3, 1)
        img = ax1.imshow(suv_data, cmap='gray')
        
        # Draw background ROI
        circle_bg = Circle((center_x, center_y), bg_radius_pixels, 
                        fill=False, edgecolor='green', linewidth=2)
        ax1.add_patch(circle_bg)
        
        # Draw cylinder ROIs
        for cylinder_id, data in cylinder_measurements.items():
            cy, cx = data['position']
            radius = data['radius_pixels']
            color = cylinder_colors.get(cylinder_id, 'gray')
            
            circle = Circle((cx, cy), radius, fill=False, edgecolor=color, linewidth=2)
            ax1.add_patch(circle)
        
         # Add slice information to the title
        title_parts = ["ACR Phantom with ROIs"]
        if 'slice_group' in results:
            num_slices = results.get('num_slices', len(results['slice_group']))
            total_thickness = results.get('total_thickness_mm', None)
            
            slice_info = f"{num_slices} Slices"
            if total_thickness is not None:
                slice_info += f" ({total_thickness:.1f} mm total)"
            
            title_parts.append(slice_info)
        elif 'slice_index' in results:
            # Single slice case
            slice_info = f"Slice {results['slice_index']}"
            if hasattr(slice_data, 'SliceThickness'):
                slice_info += f" ({slice_data.SliceThickness} mm)"
            
            title_parts.append(slice_info)
        
        # Add scan date
        title_parts.append(f"Scan Date: {scan_date}")
        
        # Join all title parts with separators
        ax1.set_title(" - ".join(title_parts))
        
        # Add colorbar to the first subplot
        cb = plt.colorbar(img, ax=ax1, pad=0.01, fraction=0.046)
        cb.set_label(value_label)
        
        # 2. Bar chart of ROI values
        ax2 = plt.subplot(2, 3, 2)
        
        # Prepare data for bar chart
        roi_names = ['Background']
        mean_values = [background_mean]
        max_values = [background_max]
        colors = ['green']
        
        # Add cylinders to bar chart
        for cylinder_id in sorted(cylinder_measurements.keys()):
            data = cylinder_measurements[cylinder_id]
            roi_names.append(data['description'])
            mean_values.append(data['mean'])
            max_values.append(data['max'])
            colors.append(cylinder_colors.get(cylinder_id, 'gray'))
        
        # Position bars for mean and max values
        x = np.arange(len(roi_names))
        width = 0.35
        
        ax2.bar(x - width/2, mean_values, width, color=colors, alpha=0.7, label='Mean')
        ax2.bar(x + width/2, max_values, width, color=colors, label='Max')
        
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)  # Target for background
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7)    # Target for 25mm cylinder
        ax2.set_ylabel(value_label)
        ax2.set_xticks(x)
        ax2.set_xticklabels(roi_names, rotation=45, ha='right')
        ax2.set_title(f'ROI Measurements ({value_label})')
        ax2.legend()
        
        # 3. ROI Measurements
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')  # Hide axes
        ax3.text(0.05, 0.95, "ROI Measurements:", fontsize=14, weight='bold', transform=ax3.transAxes)
        
        # Display background measurements
        ax3.text(0.05, 0.85, f"Background ROI (green):", color='green', fontsize=12, weight='bold', transform=ax3.transAxes)
        ax3.text(0.05, 0.80, f"  Diameter: {bg_diameter_mm} mm", color='green', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.05, 0.75, f"  Mean: {background_mean:.2f}", color='green', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.05, 0.70, f"  Max: {background_max:.2f}", color='green', fontsize=12, transform=ax3.transAxes)
        
        # Display cylinder measurements
        y_pos = 0.60
        for cylinder_id in sorted(cylinder_measurements.keys()):
            data = cylinder_measurements[cylinder_id]
            color = cylinder_colors.get(cylinder_id, 'gray')
            
            ax3.text(0.05, y_pos, f"{data['description']} ({color}):", color=color, fontsize=12, weight='bold', transform=ax3.transAxes)
            ax3.text(0.05, y_pos-0.05, f"  Diameter: {data['diameter_mm']} mm", color=color, fontsize=12, transform=ax3.transAxes)
            ax3.text(0.05, y_pos-0.10, f"  Mean: {data['mean']:.2f}", color=color, fontsize=12, transform=ax3.transAxes)
            ax3.text(0.05, y_pos-0.15, f"  Max: {data['max']:.2f}", color=color, fontsize=12, transform=ax3.transAxes)
            
            y_pos -= 0.25
        
        # 4. Analysis Results
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')  # Hide axes
        ax4.text(0.05, 0.95, "Analysis Results (Mixed Criteria):", fontsize=14, weight='bold', transform=ax4.transAxes)
        ax4.text(0.05, 0.9, "Background: SUVmean, Cylinders: SUVmax", fontsize=12, style='italic', transform=ax4.transAxes)
        
        # Display pass/fail criteria
        ax4.text(0.05, 0.8, f"Background Target: 0.90 – 1.10", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.05, 0.75, f"Result: {background_mean:.2f} (Mean) ({'PASS' if results['background_pass'] else 'FAIL'})", 
                color='green' if results['background_pass'] else 'red', fontsize=12, transform=ax4.transAxes)
        
        if 'cylinder_25mm_pass' in results:
            ax4.text(0.05, 0.65, f"25mm Cylinder Target: 1.87 – 2.91", fontsize=12, transform=ax4.transAxes)
            ax4.text(0.05, 0.6, f"Result: {results['cylinder_25mm_max']:.2f} (Max) ({'PASS' if results['cylinder_25mm_pass'] else 'FAIL'})", 
                    color='green' if results['cylinder_25mm_pass'] else 'red', fontsize=12, transform=ax4.transAxes)
        
        if 'ratio_16_25_pass' in results:
            ax4.text(0.05, 0.5, f"16mm/25mm Ratio Target: > 0.7", fontsize=12, transform=ax4.transAxes)
            ax4.text(0.05, 0.45, f"Result: {results['ratio_16_25_max']:.2f} (Max/Max) ({'PASS' if results['ratio_16_25_pass'] else 'FAIL'})", 
                    color='green' if results['ratio_16_25_pass'] else 'red', fontsize=12, transform=ax4.transAxes)
        
        # 5. Ratio Analysis
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')  # Hide axes
        ax5.text(0.05, 0.95, "Cylinder Ratios:", fontsize=14, weight='bold', transform=ax5.transAxes)
        
        # Display ratios between cylinders
        y_pos = 0.85
        for cylinder_id in sorted(cylinder_measurements.keys()):
            if cylinder_id == 'cylinder_25mm':
                continue  # Skip the reference cylinder
                
            if 'cylinder_25mm' in cylinder_measurements:
                data = cylinder_measurements[cylinder_id]
                ref_data = cylinder_measurements['cylinder_25mm']
                color = cylinder_colors.get(cylinder_id, 'gray')
                
                ratio_mean = data['mean'] / ref_data['mean']
                ratio_max = data['max'] / ref_data['max']
                
                ax5.text(0.05, y_pos, f"{data['description']}/25mm Ratios:", color=color, fontsize=12, weight='bold', transform=ax5.transAxes)
                ax5.text(0.05, y_pos-0.05, f"  Mean Ratio: {ratio_mean:.2f}", color=color, fontsize=12, transform=ax5.transAxes)
                ax5.text(0.05, y_pos-0.10, f"  Max Ratio: {ratio_max:.2f}", color=color, fontsize=12, transform=ax5.transAxes)
                
                y_pos -= 0.20
        
        # 6. SUV Calculation Info
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')  # Hide axes
        
        # Show SUV calculation info if available
        if suv_info['suv_calculated']:
            ax6.text(0.05, 0.95, "SUV Calculation Info:", fontsize=14, weight='bold', transform=ax6.transAxes)
            ax6.text(0.05, 0.85, f"Patient Weight: {suv_info['patient_weight_kg']:.1f} kg", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.05, 0.80, f"Injected Dose: {suv_info['injected_dose_bq']/1e6:.2f} MBq", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.05, 0.75, f"Decay Time: {suv_info['decay_time_seconds']/60:.1f} min", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.05, 0.70, f"Decay Factor: {suv_info['decay_factor']:.4f}", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.05, 0.65, f"SUV Type: {suv_info['suv_type'].upper()}", fontsize=12, transform=ax6.transAxes)
        else:
            ax6.text(0.05, 0.95, f"Using {value_label}s (SUV calculation unavailable)", fontsize=14, weight='bold', transform=ax6.transAxes)
            ax6.text(0.05, 0.90, f"Reason: {suv_info['reason']}", fontsize=12, transform=ax6.transAxes)
            
        # Add ROI color legend
        ax6.text(0.05, 0.55, "ROI Color Legend:", fontsize=14, weight='bold', transform=ax6.transAxes)
        
        # Create legend entries
        ax6.text(0.05, 0.45, "■", color='green', fontsize=20, transform=ax6.transAxes)
        ax6.text(0.15, 0.45, f"Background ROI ({bg_diameter_mm}mm)", fontsize=12, transform=ax6.transAxes)
        
        y_pos = 0.35
        for cylinder_id, color in cylinder_colors.items():
            if cylinder_id in cylinder_measurements:
                data = cylinder_measurements[cylinder_id]
                ax6.text(0.05, y_pos, "■", color=color, fontsize=20, transform=ax6.transAxes)
                ax6.text(0.15, y_pos, f"{data['description']} ({data['diameter_mm']}mm)", fontsize=12, transform=ax6.transAxes)
                y_pos -= 0.10
        
        plt.suptitle(f"ACR Phantom Analysis - Scan Date: {scan_date}", fontsize=16)
        plt.subplots_adjust(top=0.92)  # Make room for the suptitle
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.show()
    
    return results, suv_data, (center_y, center_x)


def analyze_individual_slices(dicom_series, selected_indices):
    """
    Analyze each selected slice individually.
    
    Parameters:
    dicom_series (list): List of DICOM datasets
    selected_indices (list): List of slice indices to analyze
    
    Returns:
    list: Results from all analyzed slices
    """
    print(f"Analyzing {len(selected_indices)} selected slices: {selected_indices}")
    
    # Analyze each selected slice
    all_results = []
    for slice_idx in selected_indices:
        try:
            print(f"\nAnalyzing slice {slice_idx}...")
            # Don't display figures for individual slices (display_figures=False)
            results, image, center = analyze_acr_phantom([dicom_series[slice_idx]], 
                                             display_figures=False, 
                                             slice_index=slice_idx)
            
            
            # Store results with slice index
            results['slice_index'] = slice_idx
            all_results.append(results)
            
            # Print results for this slice
            print(f"Slice {slice_idx} Analysis Results:")
            print(f"  Background: Mean={results['background_mean']:.2f}, Max={results['background_max']:.2f} {'PASS' if results.get('background_pass') else 'FAIL'}")
            print(f"  25mm Cylinder: Mean={results['cylinder_25mm_mean']:.2f}, Max={results['cylinder_25mm_max']:.2f} {'PASS' if results.get('cylinder_25mm_pass') else 'FAIL'}")
            
            if results.get('ratio_16_25') is not None:
                print(f"  16mm/25mm Ratio: {results['ratio_16_25']:.2f} {'PASS' if results.get('ratio_16_25_pass') else 'FAIL'}")
            else:
                print("  16mm/25mm Ratio: Not calculated (16mm cylinder not detected)")
            
        except Exception as e:
            print(f"Error analyzing slice {slice_idx}: {e}")
    
    return all_results

def main(dicom_directory):
    """
    Main function to run the ACR phantom SUV analysis with interactive slice selection.
    
    Parameters:
    dicom_directory (str): Path to the directory containing DICOM files
    """
    # Read DICOM series
    dicom_series = read_dicom_series(dicom_directory)
    if not dicom_series:
        print("No DICOM files found in the directory.")
        return
    
    print(f"Found {len(dicom_series)} DICOM files.")
    
    # Display previews and let user select slices
    print("Displaying slice previews. Please select which slices to analyze.")
    selected_indices = display_slice_previews(dicom_series)
    
    if not selected_indices:
        print("No slices selected for analysis.")
        return
    
    # Ask if user wants to sum slices for 1cm thickness
    slices_for_1cm = calculate_slices_for_1cm_thickness(dicom_series)
    if slices_for_1cm and slices_for_1cm > 1:
        sum_option = input(f"\nDo you want to sum consecutive slices to achieve 1cm thickness (approximately {slices_for_1cm} slices)? (y/n): ")
        if sum_option.lower().startswith('y'):
            # Group selected indices into consecutive groups
            selected_groups = []
            current_group = []
            
            # Sort indices first
            sorted_indices = sorted(selected_indices)
            
            for idx in sorted_indices:
                if not current_group or idx == current_group[-1] + 1:
                    current_group.append(idx)
                    if len(current_group) == slices_for_1cm:
                        selected_groups.append(current_group)
                        current_group = []
                else:
                    if current_group:  # Add any remaining partial group
                        selected_groups.append(current_group)
                    current_group = [idx]
            
            if current_group:  # Add any final partial group
                selected_groups.append(current_group)
            
            print("\nSlice groups for analysis:")
            for i, group in enumerate(selected_groups):
                print(f"Group {i+1}: Slices {group} ({len(group)} slices)")
            
            # Analyze each group by summing the pixel data
            all_results = []
            for group in selected_groups:
                try:
                    print(f"\nAnalyzing group with slices {group}...")
                    
                    # Sum the pixel data from all slices in this group
                    summed_slice = dicom_series[group[0]].copy()
                    summed_pixel_data = summed_slice.pixel_array.astype(float)
                    
                    for idx in group[1:]:
                        summed_pixel_data += dicom_series[idx].pixel_array.astype(float)
                    
                    # Average the summed data (optional, depends on your protocol)
                    summed_pixel_data /= len(group)
                    
                    # Replace pixel data in the first slice with summed data
                    summed_slice.PixelData = summed_pixel_data.astype(np.uint16).tobytes()
                    

                    # Store slice information
                    num_slices = len(group)
                    total_thickness_mm = None

                    # Calculate the total thickness
                    if num_slices > 1 and hasattr(dicom_series[0], 'SliceThickness'):
                        slice_thickness = float(dicom_series[0].SliceThickness)
                        total_thickness_mm = slice_thickness * num_slices
                    elif num_slices > 1:
                        # Try to calculate from position information
                        try:
                            first_slice = dicom_series[group[0]]
                            last_slice = dicom_series[group[-1]]
                            if hasattr(first_slice, 'ImagePositionPatient') and hasattr(last_slice, 'ImagePositionPatient'):
                                z1 = first_slice.ImagePositionPatient[2]
                                z2 = last_slice.ImagePositionPatient[2]
                                total_thickness_mm = abs(z2 - z1) + float(dicom_series[0].SliceThickness)
                        except:
                            # If we can't calculate thickness, leave it as None
                            pass

                    # Analyze the summed slice with slice information
                    results, image, center = analyze_acr_phantom([summed_slice], display_figures=True, 
                                                                slice_group=group, 
                                                                num_slices=num_slices,
                                                                total_thickness_mm=total_thickness_mm)
                    
                    if 'slice_group' in results:
                        num_slices = results.get('num_slices', len(results['slice_group']))
                        total_thickness = results.get('total_thickness_mm', None)
                        
                        slice_info = f" - {num_slices} Slices"
                        if total_thickness is not None:
                            slice_info += f" ({total_thickness:.1f} mm total)"
                        
                        # Add this information to the figure title
                        ax1.set_title(f'ACR Phantom with ROIs{slice_info}')
                    
                    # Store results with slice group info
                    results['slice_group'] = group
                    all_results.append(results)
                    
                    # Print results for this group
                    print(f"Group with slices {group} Analysis Results:")
                    print(f"  Background: Mean={results['background_mean']:.2f}, Max={results['background_max']:.2f} {'PASS' if results.get('background_pass') else 'FAIL'}")
                    print(f"  25mm Cylinder: Mean={results['cylinder_25mm_mean']:.2f}, Max={results['cylinder_25mm_max']:.2f} {'PASS' if results.get('cylinder_25mm_pass') else 'FAIL'}")
                    
                    if results.get('ratio_16_25') is not None:
                        print(f"  16mm/25mm Ratio: {results['ratio_16_25']:.2f} {'PASS' if results.get('ratio_16_25_pass') else 'FAIL'}")
                    else:
                        print("  16mm/25mm Ratio: Not calculated (16mm cylinder not detected)")
                    
                except Exception as e:
                    print(f"Error analyzing group {group}: {e}")
        else:
            # Analyze each selected slice individually without figures
            all_results = analyze_individual_slices(dicom_series, selected_indices)
            
            # After analyzing all individual slices, find the best one and display it with figures
            if all_results:
                # Find the best slice (most passing criteria)
                def count_passing(result):
                    count = 0
                    if result.get('background_pass') == True:
                        count += 1
                    if result.get('cylinder_25mm_pass') == True:
                        count += 1
                    if result.get('ratio_16_25_pass') == True:
                        count += 1
                    return count
                
                best_slice_idx = max(range(len(all_results)), key=lambda i: count_passing(all_results[i]))
                best_slice = all_results[best_slice_idx]
                best_index = best_slice['slice_index']
                
                print(f"\nDisplaying best individual slice (Slice {best_index})...")
                analyze_acr_phantom([dicom_series[best_index]], display_figures=True)
    else:
        # Analyze each selected slice individually without figures
        all_results = analyze_individual_slices(dicom_series, selected_indices)
        
        # After analyzing all individual slices, find the best one and display it with figures
        if all_results:
            # Find the best slice (most passing criteria)
            def count_passing(result):
                count = 0
                if result.get('background_pass') == True:
                    count += 1
                if result.get('cylinder_25mm_pass') == True:
                    count += 1
                if result.get('ratio_16_25_pass') == True:
                    count += 1
                return count
            
            best_slice_idx = max(range(len(all_results)), key=lambda i: count_passing(all_results[i]))
            best_slice = all_results[best_slice_idx]
            best_index = best_slice['slice_index']
            
            print(f"\nDisplaying best individual slice (Slice {best_index})...")
            analyze_acr_phantom([dicom_series[best_index]], display_figures=True)
    
    # Summarize results across all analyzed slices/groups
    if all_results:
        print("\n" + "="*50)
        print("SUMMARY: ACR Phantom Analysis Report")
        print("="*50)
        
        # Find the best slice/group (most passing criteria)
        # Handle cases where some values might be None
        def count_passing(result):
            count = 0
            if result.get('background_pass') == True:
                count += 1
            if result.get('cylinder_25mm_pass') == True:
                count += 1
            if result.get('ratio_16_25_pass') == True:
                count += 1
            return count
        
        best_slice_idx = max(range(len(all_results)), key=lambda i: count_passing(all_results[i]))
        best_slice = all_results[best_slice_idx]
        
        if 'slice_index' in best_slice:
            print(f"Best results from slice {best_slice['slice_index']}:")
        else:
            print(f"Best results from slice group {best_slice['slice_group']}:")
        
        print(f"Background: Mean={best_slice['background_mean']:.2f}, Max={best_slice['background_max']:.2f} {'PASS' if best_slice.get('background_pass') else 'FAIL'}")
        print(f"25mm Cylinder: Mean={best_slice['cylinder_25mm_mean']:.2f}, Max={best_slice['cylinder_25mm_max']:.2f} {'PASS' if best_slice.get('cylinder_25mm_pass') else 'FAIL'}")
        
        if best_slice.get('ratio_16_25') is not None:
            print(f"16mm/25mm Ratio: {best_slice['ratio_16_25']:.2f} {'PASS' if best_slice.get('ratio_16_25_pass') else 'FAIL'}")
        else:
            print("16mm/25mm Ratio: Not calculated (16mm cylinder not detected)")
        
        # Count how many criteria passed
        passing = sum([
            best_slice.get('background_pass') == True,
            best_slice.get('cylinder_25mm_pass') == True,
            best_slice.get('ratio_16_25_pass') == True
        ])
        total = 2 if best_slice.get('ratio_16_25') is None else 3
        
        print(f"Overall: {passing}/{total} criteria passed")
    else:
        print("\nNo valid results obtained from any slice/group.")

if __name__ == "__main__":
    # Hardcode your DICOM directory path here
    #dicom_directory = r"P:\My_Projects\ACR\omni\012025"
    dicom_directory = r"./012025"
    
    # Run the analysis
    main(dicom_directory)