% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-search - Search the Docker Hub for images

# SYNOPSIS
**docker search**
[**--automated**[=*false*]]
[**--help**]
[**--no-trunc**[=*false*]]
[**-s**|**--stars**[=*0*]]
TERM

# DESCRIPTION

Search Docker Hub for images that match the specified `TERM`. The table
of images returned displays the name, description (truncated by default), number
of stars awarded, whether the image is official, and whether it is automated.

*Note* - Search queries will only return up to 25 results

# OPTIONS
**--automated**=*true*|*false*
   Only show automated builds. The default is *false*.

**--help**
  Print usage statement

**--no-trunc**=*true*|*false*
   Don't truncate output. The default is *false*.

**-s**, **--stars**=0
   Only displays with at least x stars

# EXAMPLES

## Search Docker Hub for ranked images

Search a registry for the term 'fedora' and only display those images
ranked 3 or higher:

    $ docker search -s 3 fedora
    NAME                  DESCRIPTION                                    STARS OFFICIAL  AUTOMATED
    mattdm/fedora         A basic Fedora image corresponding roughly...  50
    fedora                (Semi) Official Fedora base image.             38
    mattdm/fedora-small   A small Fedora image on which to build. Co...  8
    goldmann/wildfly      A WildFly application server running on a ...  3               [OK]

## Search Docker Hub for automated images

Search Docker Hub for the term 'fedora' and only display automated images
ranked 1 or higher:

    $ docker search -s 1 fedora
    NAME               DESCRIPTION                                     STARS OFFICIAL  AUTOMATED
    goldmann/wildfly   A WildFly application server running on a ...   3               [OK]
    tutum/fedora-20    Fedora 20 image with SSH access. For the r...   1               [OK]

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
April 2015, updated by Mary Anthony for v2 <mary@docker.com>

