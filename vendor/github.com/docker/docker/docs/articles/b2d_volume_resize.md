<!--[metadata]>
+++
title = "Resizing a Boot2Docker volume	"
description = "Resizing a Boot2Docker volume in VirtualBox with GParted"
keywords = ["boot2docker, volume,  virtualbox"]
[menu.main]
parent = "smn_win_osx"
+++
<![end-metadata]-->

# Getting “no space left on device” errors with Boot2Docker?

If you're using Boot2Docker with a large number of images, or the images you're
working with are very large, your pulls might start failing with "no space left
on device" errors when the Boot2Docker volume fills up. There are two solutions
you can try.

## Solution 1: Add the `DiskImage` property in boot2docker profile

The `boot2docker` command reads its configuration from the `$BOOT2DOCKER_PROFILE` if set, or `$BOOT2DOCKER_DIR/profile` or `$HOME/.boot2docker/profile` (on Windows this is `%USERPROFILE%/.boot2docker/profile`).

1. View the existing configuration, use the `boot2docker config` command.

        $ boot2docker config
        # boot2docker profile filename: /Users/mary/.boot2docker/profile
        Init = false
        Verbose = false
        Driver = "virtualbox"
        Clobber = true
        ForceUpgradeDownload = false
        SSH = "ssh"
        SSHGen = "ssh-keygen"
        SSHKey = "/Users/mary/.ssh/id_boot2docker"
        VM = "boot2docker-vm"
        Dir = "/Users/mary/.boot2docker"
        ISOURL = "https://api.github.com/repos/boot2docker/boot2docker/releases"
        ISO = "/Users/mary/.boot2docker/boot2docker.iso"
        DiskSize = 20000
        Memory = 2048
        CPUs = 8
        SSHPort = 2022
        DockerPort = 0
        HostIP = "192.168.59.3"
        DHCPIP = "192.168.59.99"
        NetMask = [255, 255, 255, 0]
        LowerIP = "192.168.59.103"
        UpperIP = "192.168.59.254"
        DHCPEnabled = true
        Serial = false
        SerialFile = "/Users/mary/.boot2docker/boot2docker-vm.sock"
        Waittime = 300
        Retries = 75

  The configuration shows you where `boot2docker` is looking for the `profile` file. It also output the settings that are in use.


2. Initialise a default file to customize using `boot2docker config > ~/.boot2docker/profile` command.

3. Add the following lines to `$HOME/.boot2docker/profile`:

        # Disk image size in MB
        DiskSize = 50000

4. Run the following sequence of commands to restart Boot2Docker with the new settings.

        $ boot2docker poweroff
        $ boot2docker destroy
        $ boot2docker init
        $ boot2docker up

## Solution 2: Increase the size of boot2docker volume

This solution increases the volume size by first cloning it, then resizing it
using a disk partitioning tool. We recommend
[GParted](http://gparted.sourceforge.net/download.php/index.php). The tool comes
as a bootable ISO, is a free download, and works well with VirtualBox.

1. Stop Boot2Docker

  Issue the command to stop the Boot2Docker VM on the command line:

      $ boot2docker stop

2. Clone the VMDK image to a VDI image

  Boot2Docker ships with a VMDK image, which can't be resized by VirtualBox's
  native tools. We will instead create a VDI volume and clone the VMDK volume to
  it.

3. Using the command line VirtualBox tools, clone the VMDK image to a VDI image:

        $ vboxmanage clonehd /full/path/to/boot2docker-hd.vmdk /full/path/to/<newVDIimage>.vdi --format VDI --variant Standard

4. Resize the VDI volume

  Choose a size that will be appropriate for your needs. If you're spinning up a
  lot of containers, or your containers are particularly large, larger will be
  better:

      $ vboxmanage modifyhd /full/path/to/<newVDIimage>.vdi --resize <size in MB>

5. Download a disk partitioning tool ISO

  To resize the volume, we'll use [GParted](http://gparted.sourceforge.net/download.php/).
  Once you've downloaded the tool, add the ISO to the Boot2Docker VM IDE bus.
  You might need to create the bus before you can add the ISO.

  > **Note:**
  > It's important that you choose a partitioning tool that is available as an ISO so
  > that the Boot2Docker VM can be booted with it.

  <table>
      <tr>
          <td><img src="/articles/b2d_volume_images/add_new_controller.png"><br><br></td>
      </tr>
      <tr>
          <td><img src="/articles/b2d_volume_images/add_cd.png"></td>
      </tr>
  </table>

6. Add the new VDI image

  In the settings for the Boot2Docker image in VirtualBox, remove the VMDK image
  from the SATA controller and add the VDI image.

  <img src="/articles/b2d_volume_images/add_volume.png">

7. Verify the boot order

    In the **System** settings for the Boot2Docker VM, make sure that **CD/DVD** is
    at the top of the **Boot Order** list.

    <img src="/articles/b2d_volume_images/boot_order.png">

8. Boot to the disk partitioning ISO

  Manually start the Boot2Docker VM in VirtualBox, and the disk partitioning ISO
  should start up. Using GParted, choose the **GParted Live (default settings)**
  option. Choose the default keyboard, language, and XWindows settings, and the
  GParted tool will start up and display the VDI volume you created. Right click
  on the VDI and choose **Resize/Move**.

  <img src="/articles/b2d_volume_images/gparted.png">

9. Drag the slider representing the volume to the maximum available size.

10. Click **Resize/Move** followed by **Apply**.

  <img src="/articles/b2d_volume_images/gparted2.png">

11. Quit GParted and shut down the VM.

12. Remove the GParted ISO from the IDE controller for the Boot2Docker VM in
VirtualBox.

13. Start the Boot2Docker VM

  Fire up the Boot2Docker VM manually in VirtualBox. The VM should log in
  automatically, but if it doesn't, the credentials are `docker/tcuser`. Using
  the `df -h` command, verify that your changes took effect.

  <img src="/articles/b2d_volume_images/verify.png">

You're done!
