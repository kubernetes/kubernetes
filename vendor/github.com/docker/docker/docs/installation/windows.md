<!--[metadata]>
+++
title = "Installation on Windows"
description = "Docker installation on Microsoft Windows"
keywords = ["Docker, Docker documentation, Windows, requirements, virtualbox,  boot2docker"]
[menu.main]
parent = "smn_engine"
+++
<![end-metadata]-->

# Windows
> **Note:**
> Docker has been tested on Windows 7 and 8.1; it may also run on older versions.
> Your processor needs to support hardware virtualization.

The Docker Engine uses Linux-specific kernel features, so to run it on Windows
we need to use a lightweight virtual machine (VM).  You use the **Windows Docker
Client** to control the virtualized Docker Engine to build, run, and manage
Docker containers.

To make this process easier, we've designed a helper application called
[Boot2Docker](https://github.com/boot2docker/boot2docker) which creates a Linux virtual
machine on Windows to run Docker on a Linux operating system.

Although you will be using Windows Docker client, the docker engine hosting the
containers will still be running on Linux. Until the Docker engine for Windows
is developed, you can launch only Linux containers from your Windows machine.

![Windows Architecture Diagram](/installation/images/win_docker_host.svg)

## Demonstration

<iframe width="640" height="480" src="//www.youtube.com/embed/TjMU3bDX4vo?rel=0" frameborder="0" allowfullscreen></iframe>

## Installation

1. Download the latest release of the
   [Docker for Windows Installer](https://github.com/boot2docker/windows-installer/releases/latest).
2. Run the installer, which will install Docker Client for Windows, VirtualBox,
   Git for Windows (MSYS-git), the boot2docker Linux ISO, and the Boot2Docker
   management tool.
   ![](/installation/images/windows-installer.png)
3. Run the **Boot2Docker Start** shortcut from your Desktop or “Program Files →
   Boot2Docker for Windows”.
   The Start script will ask you to enter an ssh key passphrase - the simplest
   (but least secure) is to just hit [Enter].

4. The **Boot2Docker Start** will start a unix shell already configured to manage
   Docker running inside the virtual machine. Run `docker version` to see
   if it is working correctly:

![](/installation/images/windows-boot2docker-start.png)

## Running Docker

> **Note:** if you are using a remote Docker daemon, such as Boot2Docker, 
> then _do not_ type the `sudo` before the `docker` commands shown in the
> documentation's examples.

**Boot2Docker Start** will automatically start a shell with environment variables
correctly set so you can start using Docker right away:

Let's try the `hello-world` example image. Run

    $ docker run hello-world

This should download the very small `hello-world` image and print a
`Hello from Docker.` message.

## Using Docker from Windows Command Line Prompt (cmd.exe)

Launch a Windows Command Line Prompt (cmd.exe).

Boot2Docker command requires `ssh.exe` to be in the PATH, therefore we need to
include `bin` folder of the Git installation (which has ssh.exe) to the `%PATH%`
environment variable by running:

    set PATH=%PATH%;"c:\Program Files (x86)\Git\bin"

and then we can run the `boot2docker start` command to start the Boot2Docker VM.
(Run `boot2docker init` command if you get an error saying machine does not
exist.) Then copy the instructions for cmd.exe to set the environment variables
to your console window and you are ready to run docker commands such as
`docker ps`:

![](/installation/images/windows-boot2docker-cmd.png)

## Using Docker from PowerShell

Launch a PowerShell window, then add `ssh.exe` to your PATH:

    $Env:Path = "${Env:Path};c:\Program Files (x86)\Git\bin"

and after running the `boot2docker start` command it will print PowerShell
commands to set the environment variables to connect to the Docker daemon
running inside the VM. Run these commands and you are ready to run docker
commands such as `docker ps`:

![](/installation/images/windows-boot2docker-powershell.png)

> NOTE: You can alternatively run `boot2docker shellinit | Invoke-Expression`
> command to set the environment variables instead of copying and pasting on
> PowerShell.

# Further Details

The Boot2Docker management tool provides several commands:

    $ boot2docker
    Usage: boot2docker.exe [<options>] {help|init|up|ssh|save|down|poweroff|reset|restart|config|status|info|ip|shellinit|delete|download|upgrade|version} [<args>]

## Upgrading

1. Download the latest release of the [Docker for Windows Installer](
   https://github.com/boot2docker/windows-installer/releases/latest)

2. Run the installer, which will update the Boot2Docker management tool.

3. To upgrade your existing virtual machine, open a terminal and run:

        boot2docker stop
        boot2docker download
        boot2docker start

## Container port redirection

If you are curious, the username for the boot2docker default user is `docker`
and the password is `tcuser`.

The latest version of `boot2docker` sets up a host only network adaptor which
provides access to the container's ports.

If you run a container with an exposed port:

    docker run --rm -i -t -p 80:80 nginx

Then you should be able to access that nginx server using the IP address reported
to you using:

    boot2docker ip

Typically, it is 192.168.59.103, but it could get changed by VirtualBox's DHCP
implementation.

For further information or to report issues, please see the [Boot2Docker site](http://boot2docker.io)

## Login with PUTTY instead of using the CMD

Boot2Docker generates and uses the public/private key pair in your `%USERPROFILE%\.ssh`
directory so to log in you need to use the private key from this same directory.

The private key needs to be converted into the format PuTTY uses.

You can do this with
[puttygen](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html):

- Open `puttygen.exe` and load ("File"->"Load" menu) the private key from
  `%USERPROFILE%\.ssh\id_boot2docker`
- then click: "Save Private Key".
- Then use the saved file to login with PuTTY using `docker@127.0.0.1:2022`.

## Uninstallation

You can uninstall Boot2Docker using Window's standard process for removing programs.
This process does not remove the `docker-install.exe` file. You must delete that file
yourself.

## References

If you have Docker hosts running and if you don't wish to do a 
Boot2Docker installation, you can install the docker.exe using
unofficial Windows package manager Chocolately. For information
on how to do this, see [Docker package on Chocolatey](http://chocolatey.org/packages/docker).
