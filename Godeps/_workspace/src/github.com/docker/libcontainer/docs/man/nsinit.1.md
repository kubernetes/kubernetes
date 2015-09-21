% nsinit User Manual
% docker/libcontainer
% JAN 2015

NAME:
   nsinit - A low-level utility for managing containers.
	    It is used to spawn new containers or join existing containers.

USAGE:
   nsinit [global options] command [command options] [arguments...]

VERSION:
   0.1

COMMANDS:
	config	display the container configuration 
	exec	execute a new command inside a container
	init	runs the init process inside the namespace
	oom	display oom notifications for a container
	pause	pause the container's processes
	stats	display statistics for the container
	unpause	unpause the container's processes
	help, h	shows a list of commands or help for one command

EXAMPLES:

Get the <container_id> of an already running docker container.
`sudo docker ps` will return the list of all the running containers.

take the <container_id> (e.g. 4addb0b2d307) and go to its config directory
`/var/lib/docker/execdriver/native/4addb0b2d307` and here you can run the nsinit
command line utility.

e.g. `nsinit exec /bin/bash` will start a shell on the already running container.
   
# HISTORY
Jan 2015, Originally compiled by Shishir Mahajan (shishir dot mahajan at redhat dot com)
based on nsinit source material and internal work.	
