This directory contains an example of the DaemonSet Flexvolume driver deployment method. See documentation [here](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/storage/flexvolume-deployment.md#recommended-driver-deployment-method).

Steps to use the DaemonSet deployment method:
1. Copy the Flexvolume driver to `drivers` directory. To get a basic example running, copy the `dummy` driver from the parent directory.
1. If you'd like to just get a basic example running, you could skip this step. Otherwise, change the places marked with `TODO` in all files.
1. Build the deployment Docker image and upload to your container registry.
1. Create the DaemonSet.
