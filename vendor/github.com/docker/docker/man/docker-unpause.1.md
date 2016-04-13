% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-unpause - Unpause all processes within a container

# SYNOPSIS
**docker unpause**
CONTAINER [CONTAINER...]

# DESCRIPTION

The `docker unpause` command uses the cgroups freezer to un-suspend all
processes in a container.

See the [cgroups freezer documentation]
(https://www.kernel.org/doc/Documentation/cgroups/freezer-subsystem.txt) for
further details.

# OPTIONS
There are no available options.

# See also
**docker-pause(1)** to pause all processes within a container.

# HISTORY
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
