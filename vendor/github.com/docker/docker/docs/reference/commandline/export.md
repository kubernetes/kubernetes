<!--[metadata]>
+++
title = "export"
description = "The export command description and usage"
keywords = ["export, file, system, container"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# export

    Usage: docker export [OPTIONS] CONTAINER
    
      Export the contents of a filesystem to a tar archive (streamed to STDOUT by default).

      -o, --output=""    Write to a file, instead of STDOUT

      Produces a tarred repository to the standard output stream.


 For example:

    $ docker export red_panda > latest.tar

   Or

    $ docker export --output="latest.tar" red_panda

The `docker export` command does not export the contents of volumes associated
with the container. If a volume is mounted on top of an existing directory in
the container, `docker export` will export the contents of the *underlying*
directory, not the contents of the volume.

Refer to [Backup, restore, or migrate data
volumes](/userguide/dockervolumes/#backup-restore-or-migrate-data-volumes) in
the user guide for examples on exporting data in a volume.
