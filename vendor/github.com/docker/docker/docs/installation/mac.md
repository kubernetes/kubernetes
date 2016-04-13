<!--[metadata]>
+++
title = "Installation on Mac OS X"
description = "Instructions for installing Docker on OS X using boot2docker."
keywords = ["Docker, Docker documentation, requirements, boot2docker, VirtualBox, SSH, Linux, OSX, OS X,  Mac"]
[menu.main]
parent = "smn_engine"
+++
<![end-metadata]-->

# Mac OS X

You can install Docker using Boot2Docker to run `docker` commands at your command-line.
Choose this installation if you are familiar with the command-line or plan to
contribute to the Docker project on GitHub.

[<img src="/installation/images/kitematic.png" alt="Download Kitematic"
style="float:right;">](https://kitematic.com/download)

Alternatively, you may want to try <a id="inlinelink" href="https://kitematic.com/"
target="_blank">Kitematic</a>, an application that lets you set up Docker and
run containers using a graphical user interface (GUI).

## Command-line Docker with Boot2Docker

Because the Docker daemon uses Linux-specific kernel features, you can't run
Docker natively in OS X. Instead, you must install the Boot2Docker application.
The application includes a VirtualBox Virtual Machine (VM), Docker itself, and the
Boot2Docker management tool.

The Boot2Docker management tool is a lightweight Linux virtual machine made
specifically to run the Docker daemon on Mac OS X. The VirtualBox VM runs
completely from RAM, is a small ~24MB download, and boots in approximately 5s.

**Requirements**

Your Mac must be running OS X 10.6 "Snow Leopard" or newer to run Boot2Docker.

### Learn the key concepts before installing

In a Docker installation on Linux, your machine is both the localhost and the
Docker host. In networking, localhost means your computer. The Docker host is
the machine on which the containers run.

On a typical Linux installation, the Docker client, the Docker daemon, and any
containers run directly on your localhost. This means you can address ports on a
Docker container using standard localhost addressing such as `localhost:8000` or
`0.0.0.0:8376`.

![Linux Architecture Diagram](/installation/images/linux_docker_host.svg)

In an OS X installation, the `docker` daemon is running inside a Linux virtual
machine provided by Boot2Docker.

![OSX Architecture Diagram](/installation/images/mac_docker_host.svg)

In OS X, the Docker host address is the address of the Linux VM.
When you start the `boot2docker` process, the VM is assigned an IP address. Under
`boot2docker` ports on a container map to ports on the VM. To see this in
practice, work through the exercises on this page.


### Installation

1. Go to the [boot2docker/osx-installer ](
   https://github.com/boot2docker/osx-installer/releases/latest) release page.

4. Download Boot2Docker by clicking `Boot2Docker-x.x.x.pkg` in the "Downloads"
   section.

3. Install Boot2Docker by double-clicking the package.

    The installer places Boot2Docker and VirtualBox in your "Applications" folder.

The installation places the `docker` and `boot2docker` binaries in your
`/usr/local/bin` directory.


## Start the Boot2Docker Application

To run a Docker container, you first start the `boot2docker` VM and then issue
`docker` commands to create, load, and manage containers. You can launch
`boot2docker` from your Applications folder or from the command line.

> **NOTE**: Boot2Docker is designed as a development tool. You should not use
>  it in production environments.

### From the Applications folder

When you launch the "Boot2Docker" application from your "Applications" folder, the
application:

* opens a terminal window

* creates a $HOME/.boot2docker directory

* creates a VirtualBox ISO and certs

* starts a VirtualBox VM running the `docker` daemon

Once the launch completes, you can run `docker` commands. A good way to verify
your setup succeeded is to run the `hello-world` container.

    $ docker run hello-world
    Unable to find image 'hello-world:latest' locally
    511136ea3c5a: Pull complete
    31cbccb51277: Pull complete
    e45a5af57b00: Pull complete
    hello-world:latest: The image you are pulling has been verified.
    Important: image verification is a tech preview feature and should not be
    relied on to provide security.
    Status: Downloaded newer image for hello-world:latest
    Hello from Docker.
    This message shows that your installation appears to be working correctly.

    To generate this message, Docker took the following steps:
    1. The Docker client contacted the Docker daemon.
    2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
       (Assuming it was not already locally available.)
    3. The Docker daemon created a new container from that image which runs the
       executable that produces the output you are currently reading.
    4. The Docker daemon streamed that output to the Docker client, which sent it
       to your terminal.

    To try something more ambitious, you can run an Ubuntu container with:
    $ docker run -it ubuntu bash

    For more examples and ideas, visit:
    http://docs.docker.com/userguide/


A more typical way to start and stop `boot2docker` is using the command line.

### From your command line

Initialize and run `boot2docker` from the command line, do the following:

1. Create a new Boot2Docker VM.

        $ boot2docker init

    This creates a new virtual machine. You only need to run this command once.

2. Start the `boot2docker` VM.

        $ boot2docker start

3. Display the environment variables for the Docker client.

        $ boot2docker shellinit
        Writing /Users/mary/.boot2docker/certs/boot2docker-vm/ca.pem
        Writing /Users/mary/.boot2docker/certs/boot2docker-vm/cert.pem
        Writing /Users/mary/.boot2docker/certs/boot2docker-vm/key.pem
            export DOCKER_HOST=tcp://192.168.59.103:2376
            export DOCKER_CERT_PATH=/Users/mary/.boot2docker/certs/boot2docker-vm
            export DOCKER_TLS_VERIFY=1

    The specific paths and address on your machine will be different.

4. To set the environment variables in your shell do the following:

        $ eval "$(boot2docker shellinit)"

    You can also set them manually by using the `export` commands `boot2docker`
    returns.

5. Run the `hello-world` container to verify your setup.

        $ docker run hello-world


## Basic Boot2Docker exercises

At this point, you should have `boot2docker` running and the `docker` client
environment initialized. To verify this, run the following commands:

    $ boot2docker status
    $ docker version

Work through this section to try some practical container tasks using `boot2docker` VM.

### Access container ports

1. Start an NGINX container on the DOCKER_HOST.

        $ docker run -d -P --name web nginx

    Normally, the `docker run` commands starts a container, runs it, and then
    exits. The `-d` flag keeps the container running in the background
    after the `docker run` command completes. The `-P` flag publishes exposed ports from the
    container to your local host; this lets you access them from your Mac.

2. Display your running container with `docker ps` command

        CONTAINER ID        IMAGE               COMMAND                CREATED             STATUS              PORTS                                           NAMES
        5fb65ff765e9        nginx:latest        "nginx -g 'daemon of   3 minutes ago       Up 3 minutes        0.0.0.0:49156->443/tcp, 0.0.0.0:49157->80/tcp   web  

    At this point, you can see `nginx` is running as a daemon.

3. View just the container's ports.

        $ docker port web
        443/tcp -> 0.0.0.0:49156
        80/tcp -> 0.0.0.0:49157

    This tells you that the `web` container's port `80` is mapped to port
    `49157` on your Docker host.

4. Enter the `http://localhost:49157` address (`localhost` is `0.0.0.0`) in your browser:

    ![Bad Address](/installation/images/bad_host.png)

    This didn't work. The reason it doesn't work is your `DOCKER_HOST` address is
    not the localhost address (0.0.0.0) but is instead the address of the
    `boot2docker` VM.

5. Get the address of the `boot2docker` VM.

        $ boot2docker ip
        192.168.59.103

6. Enter the `http://192.168.59.103:49157` address in your browser:

    ![Correct Addressing](/installation/images/good_host.png)

    Success!

7. To stop and then remove your running `nginx` container, do the following:

        $ docker stop web
        $ docker rm web

### Mount a volume on the container

When you start `boot2docker`, it automatically shares your `/Users` directory
with the VM. You can use this share point to mount directories onto your container.
The next exercise demonstrates how to do this.

1. Change to your user `$HOME` directory.

        $ cd $HOME

2. Make a new `site` directory.

        $ mkdir site

3. Change into the `site` directory.

        $ cd site

4. Create a new `index.html` file.

        $ echo "my new site" > index.html

5. Start a new `nginx` container and replace the `html` folder with your `site` directory.

        $ docker run -d -P -v $HOME/site:/usr/share/nginx/html --name mysite nginx

6. Get the `mysite` container's port.

        $ docker port mysite
        80/tcp -> 0.0.0.0:49166
        443/tcp -> 0.0.0.0:49165

7. Open the site in a browser:

    ![My site page](/installation/images/newsite_view.png)

8. Try adding a page to your `$HOME/site` in real time.

        $ echo "This is cool" > cool.html

9. Open the new page in the browser.

    ![Cool page](/installation/images/cool_view.png)

9. Stop and then remove your running `mysite` container.

        $ docker stop mysite
        $ docker rm mysite

## Upgrade Boot2Docker

If you running Boot2Docker 1.4.1 or greater, you can upgrade Boot2Docker from
the command line. If you are running an older version, you should use the
package provided by the `boot2docker` repository.

### From the command line

To upgrade from 1.4.1 or greater, you can do this:

1. Open a terminal on your local machine.

2. Stop the `boot2docker` application.

        $ boot2docker stop

3. Run the upgrade command.

        $ boot2docker upgrade


### Use the installer

To upgrade any version of Boot2Docker, do this:

1. Open a terminal on your local machine.

2. Stop the `boot2docker` application.

        $ boot2docker stop

3. Go to the [boot2docker/osx-installer ](
   https://github.com/boot2docker/osx-installer/releases/latest) release page.

4. Download Boot2Docker by clicking `Boot2Docker-x.x.x.pkg` in the "Downloads"
   section.

2. Install Boot2Docker by double-clicking the package.

    The installer places Boot2Docker in your "Applications" folder.


## Uninstallation 

1. Go to the [boot2docker/osx-installer ](
   https://github.com/boot2docker/osx-installer/releases/latest) release page. 

2. Download the source code by clicking `Source code (zip)` or
   `Source code (tar.gz)` in the "Downloads" section.

3. Extract the source code.

4. Open a terminal on your local machine.

5. Change to the directory where you extracted the source code:

        $ cd <path to extracted source code>

6. Make sure the uninstall.sh script is executable:

        $ chmod +x uninstall.sh

7. Run the uninstall.sh script:

        $ ./uninstall.sh


## Learning more and acknowledgement

Use `boot2docker help` to list the full command line reference. For more
information about using SSH or SCP to access the Boot2Docker VM, see the README
at  [Boot2Docker repository](https://github.com/boot2docker/boot2docker).

Thanks to Chris Jones whose [blog](http://viget.com/extend/how-to-use-docker-on-os-x-the-missing-guide)  
inspired me to redo this page.

Continue with the [Docker User Guide](/userguide).
