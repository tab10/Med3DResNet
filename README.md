# Prediction of atherosclerosis risk with unlabeled thoracic CT scans:  deep learning vs. Agatston method

## Timothy Burt, Luben Popov, Yuan Zi

### Description

### Annotation GUI
TODO: change annotation.txt to output annotation_ID*.txt with * the patient index (e.g. LIDC-0001 ->  1)
### Data preprocessing / normalization
1. Loading the DICOM files
2. Converting the pixel values to Hounsfield Units (HU) 
3. Heart segmentation   
3. Normalization that makes sense.
4. Zero centering the scans.

Dicom is the de-facto file standard in medical imaging(CT,FMRI,etc). These files contain a lot of metadata (such as the pixel size, so how long one pixel is in every dimension in the real world).

This pixel size/coarseness of the scan differs from scan to scan (e.g. the distance between slices may differ), which can hurt performance of CNN approaches. We can deal with this by isomorphic resampling, which we will do later.

Below is code to load a scan, which consists of multiple slices, which we simply save in a Python list. Every folder in the dataset is one scan (so one patient). One metadata field is missing, the pixel size in the Z direction, which is the slice thickness. Fortunately we can infer this, and we add this to the metadata.

#### Load the scans in given folder path
     def load_scan(path):
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            
        for s in slices:
            s.SliceThickness = slice_thickness
            
        return slices   
Change load path point to the folder contains patients' DICOM file and load data volume.

####Converting the pixel values to Hounsfield Units (HU) 


The unit of measurement in CT scans is the Hounsfield Unit (HU), which is a measure of radiodensity. CT scanners are carefully calibrated to accurately measure this. From Wikipedia:
![image](https://github.com/lpopov101/ACVProject/blob/master/images/4rlyReh.png)



Some scanners have cylindrical scanning bounds, but the output image is square. The pixels that fall outside of these bounds get the fixed value -2000. The first step is setting these values to 0, which currently corresponds to air. Next, let's go back to HU units, by multiplying with the rescale slope and adding the intercept (which are conveniently stored in the metadata of the scans!).

    def get_pixels_hu(slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)
    
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        
        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):
            
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
                
            image[slice_number] += np.int16(intercept)
        
        return np.array(image, dtype=np.int16)
Let's take a look at one of the patients.
![image](https://github.com/lpopov101/ACVProject/blob/master/images/HU_p1.png)

####Heart segmentation(Method 1)
1.extract crop points coordinate from jason file.

    def read_crop_points(file_path):
        crop_points=[]
        with open(file_path) as json_file:
    		data = json.load(json_file)
    		num_superior_slice=data['slices']
    		for p in data['slices']:
    			crop_points.append(p['bounds'])
    			print('file_name'+ p['file_name'])
    			print('crop_points: ' + str(p['bounds']))
    			#print('num_slices: ' + p['num_slices'])
    			print('')
    		num_inferior_slice = p
    	return crop_points, num_superior_slice, num_inferior_slice

2.creat mask with annotation file(each slices 4 points) and crop images.
3.Delete out of selected region vulume part.
    def make_mask(crop_points,img, display=False):
        row_size = img.shape[0]
    	col_size = img.shape[1]
    	# Create a Polygon
    
    	coords = crop_points
    	polygon = Polygon(coords)
    	mask = np.ndarray([row_size, col_size], dtype=np.int8)
    	mask[:] = 0
    	for i in range(row_size):
    		for j in range(col_size):
    			mask[i,j]=polygon.contains(Point(i, j))
    	# show image before and after mask
    	if (display):
    		fig, ax = plt.subplots(3, 2, figsize=[12, 12])
    		ax[0, 0].set_title("Original")
    		ax[0, 0].imshow(img, cmap='gray')
    		ax[0, 0].axis('off')
    		ax[0, 1].set_title("mask")
    		ax[0, 1].imshow(mask, cmap='gray')
    		ax[0, 1].axis('off')
    		ax[1, 0].set_title("after crop")
    		ax[1, 0].imshow(mask * img, cmap='gray')
    		ax[1, 0].axis('off')
    		plt.show()
    	return mask * img
####Resampling
A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725], this can be problematic for automatic analysis (e.g. using ConvNets)!

A common method of dealing with this is resampling the full dataset to a certain isotropic resolution. If we choose to resample everything to 1mm1mm1mm pixels we can use 3D convnets without worrying about learning zoom/slice thickness invariance.

Whilst this may seem like a very simple step, it has quite some edge cases due to rounding. Also, it takes quite a while.

Below code worked well for us (and deals with the edge cases):

    def resample(image, scan, new_spacing=[1,1,1]):
        # Determine current pixel spacing
        spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        
        return image, new_spacing
Please note that when you apply this, to save the new spacing! Due to rounding this may be slightly off from the desired spacing (above script picks the best possible spacing with rounding).

Let's resample our patient's pixels to an isomorphic resolution of 1 by 1 by 1 mm.

    print("Shape before resampling\t", imgs_to_process.shape)
    imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1, 1, 1])
    print("Shape after resampling\t", imgs_after_resamp.shape)
    Shape before resampling     (133, 512, 512)
    Shape after resampling     (332, 360, 360)
    
####3D plotting the scan
For visualization it is useful to be able to show a 3D image.So we will use marching cubes to create an approximate mesh for our 3D object, and plot this with matplotlib. 
    
    def plot_3d(image, threshold=-300):
        
        # Position the scan upright, 
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2,1,0)
        
        verts, faces = measure.marching_cubes(p, threshold)
    
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
    
        plt.show()
Our plot function takes a threshold argument which we can use to plot certain structures, such as all tissue or only the bones. 400 is a good threshold for showing the bones only (see Hounsfield unit table above). Let's do this!

    plot_3d(pix_resampled, 400)

####Heart segmentation(Method 2)
In order to reduce the problem space, we can segment the heart.


Machine learning algorithms work a lot better when you can narrowly define what it is looking at. One way to do this is by creating different models for different parts of a chest CT. For instance, a convolutional network for hearts would perform better than a general-purpose network for the whole chest.

Therefore, it is often useful to pre-process the image data by auto-detecting the boundaries surrounding a volume of interest.

The below code will:

Standardize the pixel value by subtracting the mean and dividing by the standard deviation
Identify the proper threshold by creating 2 KMeans clusters comparing centered on soft tissue/bone vs lung/air.
Using Erosion) and Dilation) which has the net effect of removing tiny features like pulmonary vessels or noise
Identify each distinct region as separate image labels (think the magic wand in Photoshop)
Using bounding boxes for each image label to identify which ones represent heart and which ones represent "every thing else"
Create the masks for heart fields.
Apply mask onto the original image to erase voxels outside of the lung fields.


But there's one thing we can fix, it is probably a good idea to include structures within the heart.


Anyway, when you want to use this mask, remember to first apply a dilation morphological operation on it (i.e. with a circular kernel). This expands the mask in all directions. The air + structures in the lung alone will not contain all nodules, in particular it will miss those that are stuck to the side of the lung, where they often appear! So expand the mask a little :)

This segmentation may fail for some edge cases. It relies on the fact that the air outside the patient is not connected to the air in the lungs. If the patient has a tracheostomy, this will not be the case, I do not know whether this is present in the dataset. Also, particulary noisy images (for instance due to a pacemaker in the image below) this method may also fail. Instead, the second largest air pocket in the body will be segmented. You can recognize this by checking the fraction of image that the mask corresponds to, which will be very small for this case. You can then first apply a morphological closing operation with a kernel a few mm in size to close these holes, after which it should work (or more simply, do not use the mask for this image).

pacemaker example

####Normalization
Our values currently range from -1024 to around 2000. Anything above 400 is not interesting to us, as these are simply bones with different radiodensity. A commonly used set of thresholds in the LUNA16 competition to normalize between are -1000 and 400. Here's some code you can use:

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
    def normalize(image):
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image
####Zero centering
As a final preprocessing step, it is advisory to zero center your data so that your mean value is 0. To do this you simply subtract the mean pixel value from all pixels.

To determine this mean you simply average all images in the whole dataset. If that sounds like a lot of work, we found this to be around 0.25 in the LUNA16 competition.

Warning: Do not zero center with the mean per image (like is done in some kernels on here). The CT scanners are calibrated to return accurate HU measurements. There is no such thing as an image with lower contrast or brightness like in normal pictures.

PIXEL_MEAN = 0.25

    def zero_center(image):
        image = image - PIXEL_MEAN
        return image

With these steps your images are ready for consumption by your CNN or other ML method :). You can do all these steps offline (one time and save the result), and I would advise you to do so and let it run overnight as it may take a long time.

Tip: To save storage space, don't do normalization and zero centering beforehand, but do this online (during training, just after loading). If you don't do this yet, your image are int16's, which are smaller than float32s and easier to compress as well.

If this tutorial helped you at all, please upvote it and leave a comment :)



### 3D CNN Architecture

#### Algorithm
1. Feed 256x256 images with normalized intensity into  

#### Training

#### Validation

### Results
TODO: mention correlation between lung cancer/smoking and heart disease as maybe bias
    how to discard patients under 40 years old (this test is almost always negative)
### Tutorial
TODO: build docker file
make youtube video

#### References
https://www.health.harvard.edu/heart-health/when-you-look-for-cancer-you-might-find-heart-disease
CT heart anatomy reference: https://www.youtube.com/watch?v=4pjkCFrcysk&t=216s
