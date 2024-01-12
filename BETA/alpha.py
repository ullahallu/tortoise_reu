# RASPERRY PI IMAGE TRASNFERRING SCRIPT

import os
import paramiko
from datetime import datetime

# Configuration
image_path = "/home/ullah/master/alpha/images"
image_name = "capture_{}.jpg".format(datetime.now().strftime("%Y%m%d%H%M%S"))
target_user = "ahmed"
target_ip = "192.168.4.25"
target_dir = "/Users/ahmed/Desktop/VisualStudioCodeStuff/REUMASTER/images"
ssh_key_path = "/home/ullah/.ssh/id_rsa"

# Ensure the image directory exists
os.makedirs(image_path, exist_ok=True)

# Capture the image
os.system(f"libcamera-still -o {os.path.join(image_path, image_name)}")

try:
    # Set up SCP client
    scp = paramiko.Transport((target_ip, 22))
    scp.connect(username=target_user, pkey=paramiko.RSAKey.from_private_key_file(ssh_key_path))

    # SCP transfer
    with paramiko.SFTPClient.from_transport(scp) as sftp:
        sftp.put(os.path.join(image_path, image_name), os.path.join(target_dir, image_name))

    # Close SCP connection
    scp.close()
except Exception as e:
    print(f"An error occurred during SCP transfer: {e}")


print(f"Image {image_name} transferred successfully to {target_user}@{target_ip}:{target_dir}")
