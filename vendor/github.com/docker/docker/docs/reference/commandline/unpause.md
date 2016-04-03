<!--[metadata]>
+++
title = "unpause"
description = "The unpause command description and usage"
keywords = ["cgroups, suspend, container"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# unpause

    Usage: docker unpause CONTAINER [CONTAINER...]

    Unpause all processes within a container

The `docker unpause` command uses the cgroups freezer to un-suspend all
processes in a container.

See the
[cgroups freezer documentation](https://www.kernel.org/doc/Documentation/cgroups/freezer-subsystem.txt)
for further details.
