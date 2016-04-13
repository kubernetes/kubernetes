<!--[metadata]>
+++
title = "Installation on IBM SoftLayer "
description = "Installation instructions for Docker on IBM Softlayer."
keywords = ["IBM SoftLayer, virtualization, cloud, docker, documentation,  installation"]
[menu.main]
parent = "smn_cloud"
+++
<![end-metadata]-->

# IBM SoftLayer

1. Create an [IBM SoftLayer account](
   https://www.softlayer.com/cloud-servers/).
2. Log in to the [SoftLayer Customer Portal](
   https://control.softlayer.com/).
3. From the *Devices* menu select [*Device List*](https://control.softlayer.com/devices)
4. Click *Order Devices* on the top right of the window below the menu bar.
5. Under *Virtual Server* click [*Hourly*](https://manage.softlayer.com/Sales/orderHourlyComputingInstance)
6. Create a new *SoftLayer Virtual Server Instance* (VSI) using the default
   values for all the fields and choose:

    - The desired location for *Datacenter*
    - *Ubuntu Linux 12.04 LTS Precise Pangolin - Minimal Install (64 bit)*
      for *Operating System*.

7. Click the *Continue Your Order* button at the bottom right.
8. Fill out VSI *hostname* and *domain*.
9. Insert the required *User Metadata* and place the order.
10. Then continue with the [*Ubuntu*](../ubuntulinux/#ubuntu-linux)
   instructions.

## What next?

Continue with the [User Guide](/userguide/).

