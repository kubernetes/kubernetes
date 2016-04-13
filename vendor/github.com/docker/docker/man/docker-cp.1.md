% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-cp - Copy files/folders between a container and the local filesystem.

# SYNOPSIS
**docker cp**
[**--help**]
CONTAINER:PATH LOCALPATH|-
LOCALPATH|- CONTAINER:PATH

# DESCRIPTION

In the first synopsis form, the `docker cp` utility copies the contents of
`PATH` from the filesystem of `CONTAINER` to the `LOCALPATH` (or stream as
a tar archive to `STDOUT` if `-` is specified).

In the second synopsis form, the contents of `LOCALPATH` (or a tar archive
streamed from `STDIN` if `-` is specified) are copied from the local machine to
`PATH` in the filesystem of `CONTAINER`.

You can copy to or from either a running or stopped container. The `PATH` can
be a file or directory. The `docker cp` command assumes all `CONTAINER:PATH`
values are relative to the `/` (root) directory of the container. This means
supplying the initial forward slash is optional; The command sees
`compassionate_darwin:/tmp/foo/myfile.txt` and
`compassionate_darwin:tmp/foo/myfile.txt` as identical. If a `LOCALPATH` value
is not absolute, is it considered relative to the current working directory.

Behavior is similar to the common Unix utility `cp -a` in that directories are
copied recursively with permissions preserved if possible. Ownership is set to
the user and primary group on the receiving end of the transfer. For example,
files copied to a container will be created with `UID:GID` of the root user.
Files copied to the local machine will be created with the `UID:GID` of the
user which invoked the `docker cp` command.

Assuming a path separator of `/`, a first argument of `SRC_PATH` and second
argument of `DST_PATH`, the behavior is as follows:

- `SRC_PATH` specifies a file
    - `DST_PATH` does not exist
        - the file is saved to a file created at `DST_PATH`
    - `DST_PATH` does not exist and ends with `/`
        - Error condition: the destination directory must exist.
    - `DST_PATH` exists and is a file
        - the destination is overwritten with the contents of the source file
    - `DST_PATH` exists and is a directory
        - the file is copied into this directory using the basename from
          `SRC_PATH`
- `SRC_PATH` specifies a directory
    - `DST_PATH` does not exist
        - `DST_PATH` is created as a directory and the *contents* of the source
           directory are copied into this directory
    - `DST_PATH` exists and is a file
        - Error condition: cannot copy a directory to a file
    - `DST_PATH` exists and is a directory
        - `SRC_PATH` does not end with `/.`
            - the source directory is copied into this directory
        - `SRC_PAPTH` does end with `/.`
            - the *content* of the source directory is copied into this
              directory

The command requires `SRC_PATH` and `DST_PATH` to exist according to the above
rules. If `SRC_PATH` is local and is a symbolic link, the symbolic link, not
the target, is copied.

A colon (`:`) is used as a delimiter between `CONTAINER` and `PATH`, but `:`
could also be in a valid `LOCALPATH`, like `file:name.txt`. This ambiguity is
resolved by requiring a `LOCALPATH` with a `:` to be made explicit with a
relative or absolute path, for example:

    `/path/to/file:name.txt` or `./file:name.txt`

It is not possible to copy certain system files such as resources under
`/proc`, `/sys`, `/dev`, and mounts created by the user in the container.

Using `-` as the first argument in place of a `LOCALPATH` will stream the
contents of `STDIN` as a tar archive which will be extracted to the `PATH` in
the filesystem of the destination container. In this case, `PATH` must specify
a directory.

Using `-` as the second argument in place of a `LOCALPATH` will stream the
contents of the resource from the source container as a tar archive to
`STDOUT`.

# OPTIONS
**--help**
  Print usage statement

# EXAMPLES

Suppose a container has finished producing some output as a file it saves
to somewhere in its filesystem. This could be the output of a build job or
some other computation. You can copy these outputs from the container to a
location on your local host.

If you want to copy the `/tmp/foo` directory from a container to the
existing `/tmp` directory on your host. If you run `docker cp` in your `~`
(home) directory on the local host:

		$ docker cp compassionate_darwin:tmp/foo /tmp

Docker creates a `/tmp/foo` directory on your host. Alternatively, you can omit
the leading slash in the command. If you execute this command from your home
directory:

		$ docker cp compassionate_darwin:tmp/foo tmp

If `~/tmp` does not exist, Docker will create it and copy the contents of
`/tmp/foo` from the container into this new directory. If `~/tmp` already
exists as a directory, then Docker will copy the contents of `/tmp/foo` from
the container into a directory at `~/tmp/foo`.

When copying a single file to an existing `LOCALPATH`, the `docker cp` command
will either overwrite the contents of `LOCALPATH` if it is a file or place it
into `LOCALPATH` if it is a directory, overwriting an existing file of the same
name if one exists. For example, this command:

		$ docker cp sharp_ptolemy:/tmp/foo/myfile.txt /test

If `/test` does not exist on the local machine, it will be created as a file
with the contents of `/tmp/foo/myfile.txt` from the container. If `/test`
exists as a file, it will be overwritten. Lastly, if `/tmp` exists as a
directory, the file will be copied to `/test/myfile.txt`.

Next, suppose you want to copy a file or folder into a container. For example,
this could be a configuration file or some other input to a long running
computation that you would like to place into a created container before it
starts. This is useful because it does not require the configuration file or
other input to exist in the container image.

If you have a file, `config.yml`, in the current directory on your local host
and wish to copy it to an existing directory at `/etc/my-app.d` in a container,
this command can be used:

		$ docker cp config.yml myappcontainer:/etc/my-app.d

If you have several files in a local directory `/config` which you need to copy
to a directory `/etc/my-app.d` in a container:

		$ docker cp /config/. myappcontainer:/etc/my-app.d

The above command will copy the contents of the local `/config` directory into
the directory `/etc/my-app.d` in the container.

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
May 2015, updated by Josh Hawn <josh.hawn@docker.com>
