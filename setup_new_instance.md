### Setup SSH key
Run `sudo ssh-keygen` in the vm to generate a new ssh key.

Run `cat ~/.ssh/id_rsa.pub` to get the key printed to the terminal.

Copy this key and input it in GitHub. Account Settings > SSH and GPG Keys > New SSH Key

### Config Git
Run `git config --global user.name "gcloud_instance"` (or some other name)

Run `git config --global user.email "YOUR_EMAIL"` with YOUR_EMAIL being your email (no shit)

Run `git clone git@github.com:DennisBohm1/crack_segmentation.git`.

Run `git pull` (should already be up to date though).

Run `git checkout gpu-algorithm` to switch to the right branch.

### Download dataset
Go to `https://www.station307.com/#/` and upload the zip file of the [dataset](https://drive.google.com/open?id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP).
In the vm run `wget STATION_LINK` with STATION_LINK being the link

### Run setup script
After the download of the zip file has completed, you should be good to go and run the setup script with `sh setup.sh`

### Run Experiment
You can now run the second run of the experiment with `sh run_second_experiment.sh` (Don't 'sudo' this command, cause it will somehow throw syntax errors then)