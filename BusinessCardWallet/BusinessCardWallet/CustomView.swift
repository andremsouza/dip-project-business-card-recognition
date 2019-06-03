//
//  CustomView.swift
//  BusinessCardWallet
//
//  Created by Ringa - mac mini 2 on 03/06/19.
//  Copyright © 2019 Ringa - mac mini 2. All rights reserved.
//

import UIKit

class CustomView: UIImageView {

  func showImage(_ image:UIImage?) {
    guard image != nil else { return }
    self.image = image
    contentMode = .scaleAspectFit
    isUserInteractionEnabled = true
    addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(hideImage)))
  }
  
  @objc private func hideImage(){
    UIView.animate(withDuration: 0.5, animations: {
      self.backgroundColor = .clear
      self.frame.size = CGSize(width: 0, height: 0)
    }) { (success) in
      self.removeFromSuperview()
    }
  }
  
}
