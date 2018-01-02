# NAME
   runc ps - ps displays the processes running inside a container

# SYNOPSIS
   runc ps [command options] <container-id> [ps options]

# OPTIONS
   --format value, -f value     select one of: table(default) or json

The default format is table.  The following will output the processes of a container
in json format:

    # runc ps -f json <container-id>
