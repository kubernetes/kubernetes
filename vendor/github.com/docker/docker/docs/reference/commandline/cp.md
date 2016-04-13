<!--[metadata]>
+++
title = "cp"
description = "The cp command description and usage"
keywords = ["copy, container, files, folders"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# cp

Copy files/folders between a container and the local filesystem.

    Usage:  docker cp [options] CONTAINER:PATH LOCALPATH|-
            docker cp [options] LOCALPATH|- CONTAINER:PATH

    --help  Print usage statement

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
