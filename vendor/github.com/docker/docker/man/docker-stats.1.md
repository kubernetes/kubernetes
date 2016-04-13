% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-stats - Display a live stream of one or more containers' resource usage statistics

# SYNOPSIS
**docker stats**
[**--help**]
CONTAINER [CONTAINER...]

# DESCRIPTION

Display a live stream of one or more containers' resource usage statistics

# OPTIONS
**--help**
  Print usage statement

**--no-stream**="false"
  Disable streaming stats and only pull the first result

# EXAMPLES

Run **docker stats** with multiple containers.

    $ docker stats redis1 redis2
    CONTAINER           CPU %               MEM USAGE/LIMIT     MEM %               NET I/O
    redis1              0.07%               796 KB/64 MB        1.21%               788 B/648 B
    redis2              0.07%               2.746 MB/64 MB      4.29%               1.266 KB/648 B

