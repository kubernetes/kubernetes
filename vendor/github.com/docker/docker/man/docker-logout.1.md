% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-logout - Log out from a Docker registry.

# SYNOPSIS
**docker logout**
[SERVER]

# DESCRIPTION
Log out of a Docker Registry located on the specified `SERVER`. You can
specify a URL or a `hostname` for the `SERVER` value. If you do not specify a
`SERVER`, the command attempts to log you out of Docker's public registry
located at `https://registry-1.docker.io/` by default.  

# OPTIONS
There are no available options.

# EXAMPLES

## Log out from a registry on your localhost

    # docker logout localhost:8080

# See also
**docker-login(1)** to register or log in to a Docker registry server.

# HISTORY
June 2014, Originally compiled by Daniel, Dao Quang Minh (daniel at nitrous dot io)
July 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
April 2015, updated by Mary Anthony for v2 <mary@docker.com>
