# NAME
   runc exec - execute new process inside the container

# SYNOPSIS
   runc exec [command options] <container-id> -- <container command> [args...]

Where "<container-id>" is the name for the instance of the container and
"<container command>" is the command to be executed in the container.

# EXAMPLE
For example, if the container is configured to run the linux ps command the
following will output a list of processes running in the container:

       # runc exec <container-id> ps

# OPTIONS
   --console value              specify the pty slave path for use with the container
   --cwd value                  current working directory in the container
   --env value, -e value        set environment variables
   --tty, -t                    allocate a pseudo-TTY
   --user value, -u value       UID (format: <uid>[:<gid>])
   --process value, -p value    path to the process.json
   --detach, -d                 detach from the container's process
   --pid-file value             specify the file to write the process id to
   --process-label value        set the asm process label for the process commonly used with selinux
   --apparmor value             set the apparmor profile for the process
   --no-new-privs               set the no new privileges value for the process
   --cap value, -c value        add a capability to the bounding set for the process
   --no-subreaper               disable the use of the subreaper used to reap reparented processes
