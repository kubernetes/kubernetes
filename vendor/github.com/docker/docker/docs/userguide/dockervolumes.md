<!--[metadata]>
+++
title = "Managing data in containers"
description = "How to manage data inside your Docker containers."
keywords = ["Examples, Usage, volume, docker, documentation, user guide, data,  volumes"]
[menu.main]
parent = "smn_containers"
weight = 3
+++
<![end-metadata]-->

# Managing data in containers

So far we've been introduced to some [basic Docker
concepts](/userguide/usingdocker/), seen how to work with [Docker
images](/userguide/dockerimages/) as well as learned about [networking
and links between containers](/userguide/dockerlinks/). In this section
we're going to discuss how you can manage data inside and between your
Docker containers.

We're going to look at the two primary ways you can manage data in
Docker.

* Data volumes, and
* Data volume containers.

## Data volumes

A *data volume* is a specially-designated directory within one or more
containers that bypasses the [*Union File
System*](/terms/layer/#union-file-system). Data volumes provide several 
useful features for persistent or shared data:

- Volumes are initialized when a container is created. If the container's
  base image contains data at the specified mount point, that existing data is 
  copied into the new volume upon volume initialization.
- Data volumes can be shared and reused among containers.
- Changes to a data volume are made directly.
- Changes to a data volume will not be included when you update an image.
- Data volumes persist even if the container itself is deleted.

Data volumes are designed to persist data, independent of the container's life 
cycle. Docker therefore *never* automatically delete volumes when you remove 
a container, nor will it "garbage collect" volumes that are no longer 
referenced by a container.

### Adding a data volume

You can add a data volume to a container using the `-v` flag with the
`docker create` and `docker run` command. You can use the `-v` multiple times
to mount multiple data volumes. Let's mount a single volume now in our web
application container.

    $ docker run -d -P --name web -v /webapp training/webapp python app.py

This will create a new volume inside a container at `/webapp`.

> **Note:** 
> You can also use the `VOLUME` instruction in a `Dockerfile` to add one or
> more new volumes to any container created from that image.

Docker volumes default to mount in read-write mode, but you can also set it to be mounted read-only.

    $ docker run -d -P --name web -v /opt/webapp:ro training/webapp python app.py


### Locating a volume

You can locate the volume on the host by utilizing the 'docker inspect' command.

    $ docker inspect web

The output will provide details on the container configurations including the
volumes. The output should look something similar to the following:

    ...
    "Volumes": {
        "/webapp": "/var/lib/docker/volumes/fac362...80535"
    },
    "VolumesRW": {
        "/webapp": true
    }
    ...

You will notice in the above 'Volumes' is specifying the location on the host and 
'VolumesRW' is specifying that the volume is read/write.

### Mount a host directory as a data volume

In addition to creating a volume using the `-v` flag you can also mount a
directory from your Docker daemon's host into a container.

> **Note:**
> If you are using Boot2Docker, your Docker daemon only has limited access to
> your OS X/Windows filesystem. Boot2Docker tries to auto-share your `/Users`
> (OS X) or `C:\Users` (Windows) directory - and so you can mount files or directories
> using `docker run -v /Users/<path>:/<container path> ...` (OS X) or
> `docker run -v /c/Users/<path>:/<container path ...` (Windows). All other paths
> come from the Boot2Docker virtual machine's filesystem.

    $ docker run -d -P --name web -v /src/webapp:/opt/webapp training/webapp python app.py

This will mount the host directory, `/src/webapp`, into the container at
`/opt/webapp`.

> **Note:**
> If the path `/opt/webapp` already exists inside the container's image, its
> contents will be replaced by the contents of `/src/webapp` on the host to stay
> consistent with the expected behavior of `mount`
>
> When using Boot2Docker on Windows through git bash, there might be an issue with the 
> way the source directory name is parsed. You can fix it by using a double slash at
> the beginning of the source directory name as explained in [issue #12751](https://github.com/docker/docker/issues/12751)

This is very useful for testing, for example we can
mount our source code inside the container and see our application at work as
we change the source code. The directory on the host must be specified as an
absolute path and if the directory doesn't exist Docker will automatically
create it for you.

> **Note:** 
> This is not available from a `Dockerfile` due to the portability
> and sharing purpose of built images. The host directory is, by its nature,
> host-dependent, so a host directory specified in a `Dockerfile` probably
> wouldn't work on all hosts.

Docker volumes default to mount in read-write mode, but you can also set it to be mounted read-only.

    $ docker run -d -P --name web -v /src/webapp:/opt/webapp:ro training/webapp python app.py

Here we've mounted the same `/src/webapp` directory but we've added the `ro`
option to specify that the mount should be read-only.

### Mount a host file as a data volume

The `-v` flag can also be used to mount a single file  - instead of *just* 
directories - from the host machine.

    $ docker run --rm -it -v ~/.bash_history:/.bash_history ubuntu /bin/bash

This will drop you into a bash shell in a new container, you will have your bash 
history from the host and when you exit the container, the host will have the 
history of the commands typed while in the container.

> **Note:** 
> Many tools used to edit files including `vi` and `sed --in-place` may result 
> in an inode change. Since Docker v1.1.0, this will produce an error such as
> "*sed: cannot rename ./sedKdJ9Dy: Device or resource busy*". In the case where 
> you want to edit the mounted file, it is often easiest to instead mount the 
> parent directory.

## Creating and mounting a data volume container

If you have some persistent data that you want to share between
containers, or want to use from non-persistent containers, it's best to
create a named Data Volume Container, and then to mount the data from
it.

Let's create a new named container with a volume to share.
While this container doesn't run an application, it reuses the `training/postgres`
image so that all containers are using layers in common, saving disk space.

    $ docker create -v /dbdata --name dbdata training/postgres /bin/true

You can then use the `--volumes-from` flag to mount the `/dbdata` volume in another container.

    $ docker run -d --volumes-from dbdata --name db1 training/postgres

And another:

    $ docker run -d --volumes-from dbdata --name db2 training/postgres

In this case, if the `postgres` image contained a directory called `/dbdata`
then mounting the volumes from the `dbdata` container hides the
`/dbdata` files from the `postgres` image. The result is only the files
from the `dbdata` container are visible.

You can use multiple `--volumes-from` parameters to bring together multiple data
volumes from multiple containers.

You can also extend the chain by mounting the volume that came from the
`dbdata` container in yet another container via the `db1` or `db2` containers.

    $ docker run -d --name db3 --volumes-from db1 training/postgres

If you remove containers that mount volumes, including the initial `dbdata`
container, or the subsequent containers `db1` and `db2`, the volumes will not
be deleted.  To delete the volume from disk, you must explicitly call
`docker rm -v` against the last container with a reference to the volume. This
allows you to upgrade, or effectively migrate data volumes between containers.

> **Note:** Docker will not warn you when removing a container *without* 
> providing the `-v` option to delete its volumes. If you remove containers
> without using the `-v` option, you may end up with "dangling" volumes; 
> volumes that are no longer referenced by a container.
> Dangling volumes are difficult to get rid of and can take up a large amount
> of disk space. We're working on improving volume management and you can check
> progress on this in [pull request #14214](https://github.com/docker/docker/pull/14214)

## Backup, restore, or migrate data volumes

Another useful function we can perform with volumes is use them for
backups, restores or migrations.  We do this by using the
`--volumes-from` flag to create a new container that mounts that volume,
like so:

    $ docker run --volumes-from dbdata -v $(pwd):/backup ubuntu tar cvf /backup/backup.tar /dbdata

Here we've launched a new container and mounted the volume from the
`dbdata` container. We've then mounted a local host directory as
`/backup`. Finally, we've passed a command that uses `tar` to backup the
contents of the `dbdata` volume to a `backup.tar` file inside our
`/backup` directory. When the command completes and the container stops
we'll be left with a backup of our `dbdata` volume.

You could then restore it to the same container, or another that you've made
elsewhere. Create a new container.

    $ docker run -v /dbdata --name dbdata2 ubuntu /bin/bash

Then un-tar the backup file in the new container's data volume.

    $ docker run --volumes-from dbdata2 -v $(pwd):/backup ubuntu cd /dbdata && tar xvf /backup/backup.tar

You can use the techniques above to automate backup, migration and
restore testing using your preferred tools.

# Next steps

Now we've learned a bit more about how to use Docker we're going to see how to
combine Docker with the services available on
[Docker Hub](https://hub.docker.com) including Automated Builds and private
repositories.

Go to [Working with Docker Hub](/userguide/dockerrepos).

