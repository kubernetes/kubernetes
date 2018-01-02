# NAME
   runc events - display container events such as OOM notifications, cpu, memory, and IO usage statistics

# SYNOPSIS
   runc events [command options] <container-id>

Where "<container-id>" is the name for the instance of the container.

# DESCRIPTION
   The events command displays information about the container. By default the
information is displayed once every 5 seconds.

# OPTIONS
   --interval value     set the stats collection interval (default: 5s)
   --stats              display the container's stats then exit
