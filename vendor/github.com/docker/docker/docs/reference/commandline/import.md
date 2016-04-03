<!--[metadata]>
+++
title = "import"
description = "The import command description and usage"
keywords = ["import, file, system, container"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# import

    Usage: docker import URL|- [REPOSITORY[:TAG]]

    Create an empty filesystem image and import the contents of the
	tarball (.tar, .tar.gz, .tgz, .bzip, .tar.xz, .txz) into it, then
	optionally tag it.

      -c, --change=[]     Apply specified Dockerfile instructions while importing the image

URLs must start with `http` and point to a single file archive (.tar,
.tar.gz, .tgz, .bzip, .tar.xz, or .txz) containing a root filesystem. If
you would like to import from a local directory or archive, you can use
the `-` parameter to take the data from `STDIN`.

The `--change` option will apply `Dockerfile` instructions to the image
that is created.
Supported `Dockerfile` instructions:
`CMD`|`ENTRYPOINT`|`ENV`|`EXPOSE`|`ONBUILD`|`USER`|`VOLUME`|`WORKDIR`

## Examples

**Import from a remote location:**

This will create a new untagged image.

    $ docker import http://example.com/exampleimage.tgz

**Import from a local file:**

Import to docker via pipe and `STDIN`.

    $ cat exampleimage.tgz | docker import - exampleimagelocal:new

**Import from a local directory:**

    $ sudo tar -c . | docker import - exampleimagedir

**Import from a local directory with new configurations:**

    $ sudo tar -c . | docker import --change "ENV DEBUG true" - exampleimagedir

Note the `sudo` in this example – you must preserve
the ownership of the files (especially root ownership) during the
archiving with tar. If you are not root (or the sudo command) when you
tar, then the ownerships might not get preserved.

