# NAME
   runc checkpoint - checkpoint a running container

# SYNOPSIS
   runc checkpoint [command options] <container-id>

Where "<container-id>" is the name for the instance of the container to be
checkpointed.

# DESCRIPTION
   The checkpoint command saves the state of the container instance.

# OPTIONS
   --image-path value           path for saving criu image files
   --work-path value            path for saving work files and logs
   --parent-path value          path for previous criu image files in pre-dump
   --leave-running              leave the process running after checkpointing
   --tcp-established            allow open tcp connections
   --ext-unix-sk                allow external unix sockets
   --shell-job                  allow shell jobs
   --page-server value          ADDRESS:PORT of the page server
   --file-locks                 handle file locks, for safety
   --pre-dump                   dump container's memory information only, leave the container running after this
   --manage-cgroups-mode value  cgroups mode: 'soft' (default), 'full' and 'strict'
   --empty-ns value             create a namespace, but don't restore its properties
