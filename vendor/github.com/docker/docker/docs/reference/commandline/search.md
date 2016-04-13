<!--[metadata]>
+++
title = "search"
description = "The search command description and usage"
keywords = ["search, hub, images"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# search

    Usage: docker search [OPTIONS] TERM

    Search the Docker Hub for images

      --automated=false    Only show automated builds
      --no-trunc=false     Don't truncate output
      -s, --stars=0        Only displays with at least x stars

Search [Docker Hub](https://hub.docker.com) for images

See [*Find Public Images on Docker Hub*](/userguide/dockerrepos/#searching-for-images) for
more details on finding shared images from the command line.

> **Note:**
> Search queries will only return up to 25 results

