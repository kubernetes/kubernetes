# NAME
   runc kill - kill sends the specified signal (default: SIGTERM) to the container's init process

# SYNOPSIS
   runc kill [command options] <container-id> <signal>

Where "<container-id>" is the name for the instance of the container and
"<signal>" is the signal to be sent to the init process.

# OPTIONS
   --all, -a  send the specified signal to all processes inside the container

# EXAMPLE

For example, if the container id is "ubuntu01" the following will send a "KILL"
signal to the init process of the "ubuntu01" container:

       # runc kill ubuntu01 KILL
