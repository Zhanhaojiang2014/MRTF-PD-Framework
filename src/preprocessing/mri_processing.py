import numpy as np
import nibabel as nib

from scipy.ndimage import zoom


class MRIProcessor:


    """
    MRI preprocessing pipeline:

    - Skull stripping placeholder
    - Bias correction
    - Intensity normalization
    - Resize 224x224
    """



    def __init__(
        self,
        target_size=(224,224)
    ):

        self.target_size=target_size



    def load_volume(self,path):

        img=nib.load(path)

        data=img.get_fdata()

        return data



    def skull_stripping(
        self,
        volume
    ):

        """
        Simple intensity based masking.
        In clinical pipelines this can be replaced
        by BET/FSL or deep segmentation models.
        """

        threshold=np.percentile(
            volume,
            20
        )


        mask=volume>threshold


        return volume*mask



    def bias_correction(
        self,
        volume
    ):

        mean=np.mean(volume)

        corrected=volume/(mean+1e-8)

        return corrected



    def intensity_normalization(
        self,
        volume
    ):

        return (
            volume-np.mean(volume)
        )/(np.std(volume)+1e-8)



    def resize(
        self,
        volume
    ):

        factors=(

            self.target_size[0]/volume.shape[0],

            self.target_size[1]/volume.shape[1],

            1

        )


        return zoom(
            volume,
            factors
        )



    def process(
        self,
        path
    ):

        volume=self.load_volume(path)


        volume=self.skull_stripping(
            volume
        )

        volume=self.bias_correction(
            volume
        )

        volume=self.intensity_normalization(
            volume
        )


        volume=self.resize(
            volume
        )


        return volume
