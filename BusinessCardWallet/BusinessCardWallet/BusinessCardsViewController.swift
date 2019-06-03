//
//  ViewController.swift
//  BusinessCardWallet
//
//  Created by Ringa - mac mini 2 on 28/05/19.
//  Copyright © 2019 Ringa - mac mini 2. All rights reserved.
//

import UIKit

class BusinessCardsViewController: UIViewController {

  @IBOutlet var tableView: UITableView!
  
  override func viewDidLoad() {
    super.viewDidLoad()
    
  }

}

extension BusinessCardsViewController : UITableViewDelegate, UITableViewDataSource {
  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return 0
  }
  
  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    return UITableViewCell()
  }
  
}
