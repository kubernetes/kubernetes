<!--[metadata]>
+++
title = "Work with a development container"
description = "How to use Docker's development environment"
keywords = ["development, inception, container, image Dockerfile, dependencies, Go,  artifacts"]
[menu.main]
parent = "smn_develop"
weight=5
+++
<![end-metadata]-->

# Work with a development container

In this section, you learn to develop like a member of Docker's core team.
The `docker` repository includes a `Dockerfile` at its root. This file defines
Docker's development environment.  The `Dockerfile` lists the environment's
dependencies: system libraries and binaries, Go environment, Go dependencies,
etc. 

Docker's development environment is itself, ultimately a Docker container.
You use the `docker` repository and its `Dockerfile` to create a Docker image, 
run a Docker container, and develop code in the container. Docker itself builds,
tests, and releases new Docker versions using this container.

If you followed the procedures that <a href="/project/set-up-git" target="_blank">
set up Git for contributing</a>, you should have a fork of the `docker/docker`
repository. You also created a branch called `dry-run-test`. In this section,
you continue working with your fork on this branch.

##  Clean your host of Docker artifacts

Docker developers run the latest stable release of the Docker software (with Boot2Docker if their machine is Mac OS X). They clean their local
hosts of unnecessary Docker artifacts such as stopped containers or unused
images. Cleaning unnecessary artifacts isn't strictly necessary, but it is
good practice, so it is included here.

To remove unnecessary artifacts,

1. Verify that you have no unnecessary containers running on your host.

        $ docker ps

    You should see something similar to the following:

    <table class="code">
      <tr>
        <th>CONTAINER ID</th>
        <th>IMAGE</th>
        <th>COMMAND</th>
        <th>CREATED</th>
        <th>STATUS</th>
        <th>PORTS</th>
        <th>NAMES</th>
      </tr>
    </table>

    There are no running containers on this host. If you have running but unused
    containers, stop and then remove them with the `docker stop` and `docker rm`
    commands.

2. Verify that your host has no dangling images.

        $ docker images

    You should see something similar to the following:

    <table class="code">
      <tr>
        <th>REPOSITORY</th>
        <th>TAG</th>
        <th>IMAGE ID</th>
        <th>CREATED</th>
        <th>VIRTUAL SIZE</th>
      </tr>
    </table>

    This host has no images. You may have one or more _dangling_ images. A
    dangling image is not used by a running container and is not an ancestor of
    another image on your system. A fast way to remove dangling containers is
    the following:

        $ docker rmi -f $(docker images -q -a -f dangling=true)

    This command uses `docker images` to list all images (`-a` flag) by numeric
    IDs (`-q` flag) and filter them to find dangling images (`-f dangling=true`).
    Then, the `docker rmi` command forcibly (`-f` flag) removes
    the resulting list. To remove just one image, use the `docker rmi ID`
    command.

	
## Build an image

If you followed the last procedure, your host is clean of unnecessary images 
and containers. In this section, you build an image from the Docker development
environment.

1. Open a terminal.

    Mac users, use `boot2docker status` to make sure Boot2Docker is running. You
    may need to run `eval "$(boot2docker shellinit)"` to initialize your shell
    environment.

3. Change into the root of your forked repository.

        $ cd ~/repos/docker-fork 
        
	If you are following along with this guide, you created a `dry-run-test`
	branch when you <a href="/project/set-up-git" target="_blank"> set up Git for
	contributing</a>.

4. Ensure you are on your `dry-run-test` branch.

        $ git checkout dry-run-test
        
    If you get a message that the branch doesn't exist, add the `-b` flag (git checkout -b dry-run-test) so the
    command both creates the branch and checks it out.

5. Compile your development environment container into an image.

        $ docker build -t dry-run-test .

    The `docker build` command returns informational message as it runs. The
    first build may take a few minutes to create an image. Using the
    instructions in the `Dockerfile`, the build may need to download source and
    other images. A successful build returns a final status message similar to
    the following:

        Successfully built 676815d59283

6. List your Docker images again.

        $ docker images

    You should see something similar to this:

    <table class="code">
      <tr>
        <th>REPOSTITORY</th>
        <th>TAG</th>
        <th>IMAGE ID</th>
        <th>CREATED</th>
        <th>VIRTUAL SIZE</th>
      </tr>
      <tr>
        <td>dry-run-test</td>
        <td>latest</td>
        <td>663fbee70028</td>
        <td>About a minute ago</td>
        <td></td>
      </tr>
      <tr>
        <td>ubuntu</td>
        <td>trusty</td>
        <td>2d24f826cb16</td>
        <td>2 days ago</td>
        <td>188.3 MB</td>
      </tr>
      <tr>
        <td>ubuntu</td>
        <td>trusty-20150218.1</td>
        <td>2d24f826cb16</td>
        <td>2 days ago</td>
        <td>188.3 MB</td>
      </tr>
      <tr>
        <td>ubuntu</td>
        <td>14.04</td>
        <td>2d24f826cb16</td>
        <td>2 days ago</td>
        <td>188.3 MB</td>
      </tr>
      <tr>
        <td>ubuntu</td>
        <td>14.04.2</td>
        <td>2d24f826cb16</td>
        <td>2 days ago</td>
        <td>188.3 MB</td>
      </tr>
      <tr>
        <td>ubuntu</td>
        <td>latest</td>
        <td>2d24f826cb16</td>
        <td>2 days ago</td>
        <td>188.3 MB</td>
      </tr>
    </table>

    Locate your new `dry-run-test` image in the list. You should also see a
    number of `ubuntu` images. The build process creates these. They are the
    ancestors of your new Docker development image. When you next rebuild your
    image, the build process reuses these ancestors images if they exist. 

    Keeping the ancestor images improves the build performance. When you rebuild
    the child image, the build process uses the local ancestors rather than
    retrieving them from the Hub. The build process gets new ancestors only if
    DockerHub has updated versions.

## Start a container and run a test

At this point, you have created a new Docker development environment image. Now,
you'll use this image to create a Docker container to develop in. Then, you'll
build and run a `docker` binary in your container.

1. Open two additional terminals on your host.

    At this point, you'll have about three terminals open.

    ![Multiple terminals](/project/images/three_terms.png)

    Mac OS X users, make sure you run `eval "$(boot2docker shellinit)"` in any new
    terminals.

2. In a terminal, create a new container from your `dry-run-test` image.

        $ docker run --privileged --rm -ti dry-run-test /bin/bash
        root@5f8630b873fe:/go/src/github.com/docker/docker# 

    The command creates a container from your `dry-run-test` image. It opens an
    interactive terminal (`-ti`) running a `/bin/bash` shell.  The
    `--privileged` flag gives the container access to kernel features and device
    access. This flag allows you to run a container in a container.
    Finally, the `-rm` flag instructs Docker to remove the container when you
    exit the `/bin/bash` shell.

    The container includes the source of your image repository in the
    `/go/src/github.com/docker/docker` directory. Try listing the contents to
    verify they are the same as that of your `docker-fork` repo.

    ![List example](/project/images/list_example.png)


3. Investigate your container bit. 

    If you do a `go version` you'll find the `go` language is part of the
    container. 

        root@31ed86e9ddcf:/go/src/github.com/docker/docker# go version
        go version go1.4.2 linux/amd64

    Similarly, if you do a `docker version` you find the container
    has no `docker` binary. 

        root@31ed86e9ddcf:/go/src/github.com/docker/docker# docker version
        bash: docker: command not found

    You will create one in the next steps.

4. From the `/go/src/github.com/docker/docker` directory make a `docker` binary
with the `make.sh` script.

        root@5f8630b873fe:/go/src/github.com/docker/docker# hack/make.sh binary

    You only call `hack/make.sh` to build a binary _inside_ a Docker
    development container as you are now. On your host, you'll use `make`
    commands (more about this later). 

    As it makes the binary, the `make.sh` script reports the build's progress.
    When the command completes successfully, you should see the following
    output:

	---> Making bundle: binary (in bundles/1.5.0-dev/binary)
	Created binary: /go/src/github.com/docker/docker/bundles/1.5.0-dev/binary/docker-1.5.0-dev
	
5. List all the contents of the `binary` directory.

        root@5f8630b873fe:/go/src/github.com/docker/docker#  ls bundles/1.5.0-dev/binary/
        docker  docker-1.5.0-dev  docker-1.5.0-dev.md5  docker-1.5.0-dev.sha256

    You should see that `binary` directory, just as it sounds, contains the
    made binaries.


6. Copy the `docker` binary to the `/usr/bin` of your container.

        root@5f8630b873fe:/go/src/github.com/docker/docker#  cp bundles/1.5.0-dev/binary/docker /usr/bin

7. Inside your container, check your Docker version.

        root@5f8630b873fe:/go/src/github.com/docker/docker# docker --version
        Docker version 1.5.0-dev, build 6e728fb

    Inside the container you are running a development version. This is the version
    on the current branch. It reflects the value of the `VERSION` file at the
    root of your `docker-fork` repository.

8. Start a `docker` daemon running inside your container.

        root@5f8630b873fe:/go/src/github.com/docker/docker#  docker -dD

    The `-dD` flag starts the daemon in debug mode. You'll find this useful
    when debugging your code.

9. Bring up one of the terminals on your local host.


10. List your containers and look for the container running the `dry-run-test` image.

        $ docker ps

    <table class="code">
      <tr>
        <th>CONTAINER ID</th>
        <th>IMAGE</th>
        <th>COMMAND</th>
        <th>CREATED</th>
        <th>STATUS</th>
        <th>PORTS</th>
        <th>NAMES</th>
      </tr>
      <tr>
        <td>474f07652525</td>
        <td>dry-run-test:latest</td>
        <td>"hack/dind /bin/bash</td>
        <td>14 minutes ago</td>
        <td>Up 14 minutes</td>
        <td></td>
        <td>tender_shockley</td>
      </tr>
    </table>

    In this example, the container's name is `tender_shockley`; yours will be
    different.

11. From the terminal, start another shell on your Docker development container.

        $ docker exec -it tender_shockley bash

    At this point, you have two terminals both with a shell open into your
    development container. One terminal is running a debug session. The other
    terminal is displaying a `bash` prompt.

12. At the prompt, test the Docker client by running the `hello-world` container.	

        root@9337c96e017a:/go/src/github.com/docker/docker#  docker run hello-world

    You should see the image load and return. Meanwhile, you
    can see the calls made via the debug session in your other terminal.

    ![List example](/project/images/three_running.png)


## Restart a container with your source

At this point, you have experienced the "Docker inception" technique. That is,
you have:

* built a Docker image from the Docker repository
* created and started a Docker development container from that image
* built a Docker binary inside of your Docker development container
* launched a `docker` daemon using your newly compiled binary
* called the `docker` client to run a `hello-world` container inside
  your development container

When you really get to developing code though, you'll want to iterate code
changes and builds inside the container. For that you need to mount your local
Docker repository source into your Docker container. Try that now.

1. If you haven't already, exit out of BASH shells in your running Docker
container.

    If you have followed this guide exactly, exiting out your BASH shells stops
    the running container. You can use the `docker ps` command to verify the
    development container is stopped. All of your terminals should be at the
    local host prompt.

2. Choose a terminal and make sure you are in your `docker-fork` repository.

        $ pwd
        /Users/mary/go/src/github.com/moxiegirl/docker-fork

    Your location will be different because it reflects your environment. 

3. Create a container using `dry-run-test`, but this time, mount your repository
onto the `/go` directory inside the container.

        $  docker run --privileged --rm -ti -v `pwd`:/go/src/github.com/docker/docker dry-run-test /bin/bash

    When you pass `pwd`, `docker` resolves it to your current directory.

4. From inside the container, list your `binary` directory.

        root@074626fc4b43:/go/src/github.com/docker/docker# ls bundles/1.5.0-dev/binary
        ls: cannot access binary: No such file or directory

    Your `dry-run-test` image does not retain any of the changes you made inside
    the container.  This is the expected behavior for a container. 

5. In a fresh terminal on your local host, change to the `docker-fork` root.

        $ cd ~/repos/docker-fork/

6. Create a fresh binary, but this time, use the `make` command.

        $ make BINDDIR=. binary

    The `BINDDIR` flag is only necessary on Mac OS X but it won't hurt to pass
    it on Linux command line. The `make` command, like the `make.sh` script
    inside the container, reports its progress. When the make succeeds, it
    returns the location of the new binary.


7. Back in the terminal running the container, list your `binary` directory.

        root@074626fc4b43:/go/src/github.com/docker/docker# ls bundles/1.5.0-dev/binary
        docker	docker-1.5.0-dev  docker-1.5.0-dev.md5	docker-1.5.0-dev.sha256 

    The compiled binaries created from your repository on your local host are
    now available inside your running Docker development container.

8. Repeat the steps you ran in the previous procedure.

    * copy the binary inside the development container using
      `cp bundles/1.5.0-dev/binary/docker /usr/bin`
    * start `docker -dD` to launch the Docker daemon inside the container
    * run `docker ps` on local host to get the development container's name
    * connect to your running container `docker exec -it container_name bash`
    * use the `docker run hello-world` command to create and run a container 
      inside your development container

## Where to go next

Congratulations, you have successfully achieved Docker inception. At this point,
you've set up your development environment and verified almost all the essential
processes you need to contribute. Of course, before you start contributing, 
[you'll need to learn one more piece of the development environment, the test
framework](/project/test-and-docs/).
