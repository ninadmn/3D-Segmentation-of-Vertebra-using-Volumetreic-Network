from __future__ import print_function, division
import os
import SimpleITK as sitk
import cv2
import numpy as np



def getRangImageDepth(image):
    """
    :param image:
    :return:rang of image depth
    """
    # start, end = np.where(image)[0][[0, -1]]
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition


def resize_image_itk(itkimage, newSpacing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def load_itk(filename):
    """
    load mhd files and normalization 0-255
    :param filename:
    :return:
    """
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    # Reads the image using SimpleITK
    itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itkimage


def load_itkfilewithtrucation(filename, upper=400, lower=-400):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage






def processOriginaltraindata():
    expandslice = 15
    trainImage = "/Volumes/Samsung_T5/Vert/k/images/"
    trainMask = "/Volumes/Samsung_T5/Vert/k/masks/"

    seriesindex=1
    file_list=[]
    for subsetindex in range(1,15):
        
        vertebra_subset_path = trainImage + "image00" + str(subsetindex)
        print(vertebra_subset_path)
        output_path = "/Volumes/Samsung_T5/Vert/Data1/masks/"
        vertebra_subset_mask_path = output_path
        
        file_list.append(vertebra_subset_path+".mhd")
        print(file_list)
        c=1
        for fcount in range(len(file_list)):
        
            # 1 load itk image and truncate value with upper and lower
            src = load_itkfilewithtrucation(file_list[fcount], 600, -1000)
            sub_img_file = file_list[fcount][len(vertebra_subset_path):-4]
            seg = sitk.ReadImage(vertebra_subset_mask_path + "mask00"+ str(c)+".mhd", sitk.sitkUInt8)
            c+=1
            segzspace = seg.GetSpacing()[-1]
            # 2 change z spacing >1.0 to 1.0
            if segzspace > 1.0:
                _, seg = resize_image_itk(seg, (seg.GetSpacing()[0], seg.GetSpacing()[1], 1.0),
                                              resamplemethod=sitk.sitkNearestNeighbor)
                _, src = resize_image_itk(src, (src.GetSpacing()[0], src.GetSpacing()[1], 1.0),
                                              resamplemethod=sitk.sitkLinear)
            # 3 get resample array(image and segmask)
            segimg = sitk.GetArrayFromImage(seg)
            srcimg = sitk.GetArrayFromImage(src)

            trainimagefile = trainImage + str(seriesindex)
            trainMaskfile = trainMask + str(seriesindex)
            if not os.path.exists(trainimagefile):
                os.makedirs(trainimagefile)
            if not os.path.exists(trainMaskfile):
                os.makedirs(trainMaskfile)
                # 4 get mask
            seg_liverimage = segimg.copy()
            seg_liverimage[segimg > 0] = 255
            # 5 get the roi range of mask,and expand number slices before and after,and get expand range roi image
            startpostion, endpostion = getRangImageDepth(seg_liverimage)
            if startpostion == endpostion:
                continue
            imagez = np.shape(seg_liverimage)[0]
            startpostion = startpostion - expandslice
            endpostion = endpostion + expandslice
            if startpostion < 0:
                startpostion = 0
            if endpostion > imagez:
                endpostion = imagez
            srcimg = srcimg[startpostion:endpostion, :, :]
            seg_liverimage = seg_liverimage[startpostion:endpostion, :, :]
            # 6 write src, liver mask and tumor mask image
            print(len(srcimg))
            print(len(seg_liverimage))
            for z in range(len(srcimg)):
            #print(z)
                srcimg = np.clip(srcimg, 0, 255).astype('uint8')
                cv2.imwrite(trainimagefile + "/" + str(z) + ".bmp", srcimg[z])
            for z in range(len(seg_liverimage)):
                cv2.imwrite(trainMaskfile + "/" + str(z) + ".bmp", seg_liverimage[z])
            seriesindex += 1
processOriginaltraindata()
