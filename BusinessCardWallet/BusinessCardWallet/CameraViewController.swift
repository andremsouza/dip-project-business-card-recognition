//
//  CameraViewController.swift
//  BusinessCardWallet
//
//  Created by Ringa - mac mini 2 on 28/05/19.
//  Copyright © 2019 Ringa - mac mini 2. All rights reserved.
//

import UIKit
import AVFoundation
import CropViewController

class CameraViewController: UIViewController {

  @IBOutlet var cameraView: UIView!
  var captureSession:AVCaptureSession?
  var capturePhotoOutput:AVCapturePhotoOutput?
  var previewLayer:AVCaptureVideoPreviewLayer?
  var position = AVCaptureDevice.Position.back
  var pickerController = UIImagePickerController()
  private var cardImage:UIImage?
  private var cardImageView:CustomView?
  
  override func viewDidLoad() {
    super.viewDidLoad()
    pickerController.delegate = self
    setupCamera()
  }
  
  func setupCamera(){
    previewLayer?.removeFromSuperlayer()
    captureSession = AVCaptureSession()
    guard captureSession != nil else { return }
    guard let device = AVCaptureDevice.default(AVCaptureDevice.DeviceType.builtInWideAngleCamera, for: .video, position: position) else { return }
    do {
      let input = try AVCaptureDeviceInput(device: device)
      if captureSession!.canAddInput(input) {
        captureSession!.addInput(input)
        capturePhotoOutput = AVCapturePhotoOutput()
        guard capturePhotoOutput != nil else { return }
        if captureSession!.canAddOutput(capturePhotoOutput!) {
          captureSession!.addOutput(capturePhotoOutput!)
          previewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
          previewLayer?.videoGravity = .resizeAspectFill
          previewLayer?.connection?.videoOrientation = .portrait
          previewLayer?.frame = cameraView.bounds
          guard previewLayer != nil else { return }
          cameraView.layer.addSublayer(previewLayer!)
          captureSession?.startRunning()
        }
      }
    } catch  {
      print(error.localizedDescription)
    }
  }
  
  private func setCroppedImage(_ cropController: CropViewController, with cropFrame:CGRect){
    cropController.imageCropFrame = cropFrame
    cropController.resetAspectRatioEnabled = false
    cropController.aspectRatioLockEnabled = true
    cropController.rotateButtonsHidden = true
  }
  
  @IBAction func takePicture(_ sender: UIButton) {
    guard capturePhotoOutput != nil else { return }
    let photoSettings = AVCapturePhotoSettings()
    photoSettings.previewPhotoFormat = photoSettings.embeddedThumbnailPhotoFormat
    capturePhotoOutput?.capturePhoto(with: photoSettings, delegate: self)
  }
  
  
  @IBAction func takeFromLibrary(_ sender: UIButton) {
    present(pickerController, animated: true, completion: nil)
  }
}


extension CameraViewController : UIImagePickerControllerDelegate, UINavigationControllerDelegate {
  
  func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    if let image = info[.originalImage] as? UIImage {
      let cropController = CropViewController(image: image)
      cropController.delegate = self
      //setCroppedImage(cropController, with: CGRect())
      picker.present(cropController, animated: true)
    }
  }
  
  func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
    dismiss(animated: true, completion: nil)
  }
}

extension CameraViewController : AVCapturePhotoCaptureDelegate {
  
  func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
    if let err = error {
      print(err.localizedDescription)
    }
    
    if let data = photo.fileDataRepresentation() {
      let image =  UIImage(data: data)!
      let cropController = CropViewController(image:image)
      cropController.delegate = self
      //setCroppedImage(cropController, with: CGRect())
      present(cropController, animated: true, completion: nil)
    } else {
      print("Erreur.")
    }
  }
}

extension CameraViewController : CropViewControllerDelegate {
  
  func cropViewController(_ cropViewController: CropViewController, didCropToImage image: UIImage, withRect cropRect: CGRect, angle: Int) {
    cardImage = image
    cropViewController.dismiss(animated: true) {
      //
    }
  }
}
