% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-logs - Fetch the logs of a container

# SYNOPSIS
**docker logs**
[**-f**|**--follow**[=*false*]]
[**--help**]
[**--since**[=*SINCE*]]
[**-t**|**--timestamps**[=*false*]]
[**--tail**[=*"all"*]]
CONTAINER

# DESCRIPTION
The **docker logs** command batch-retrieves whatever logs are present for
a container at the time of execution. This does not guarantee execution
order when combined with a docker run (i.e., your run may not have generated
any logs at the time you execute docker logs).

The **docker logs --follow** command combines commands **docker logs** and
**docker attach**. It will first return all logs from the beginning and
then continue streaming new output from the container’s stdout and stderr.

**Warning**: This command works only for **json-file** logging driver.

# OPTIONS
**--help**
  Print usage statement

**-f**, **--follow**=*true*|*false*
   Follow log output. The default is *false*.

**--since**=""
   Show logs since timestamp

**-t**, **--timestamps**=*true*|*false*
   Show timestamps. The default is *false*.

**--tail**="all"
   Output the specified number of lines at the end of logs (defaults to all logs)

The `--since` option shows only the container logs generated after
a given date. You can specify the date as an RFC 3339 date, a UNIX
timestamp, or a Go duration string (e.g. `1m30s`, `3h`). Docker computes
the date relative to the client machine’s time. You can combine
the `--since` option with either or both of the `--follow` or `--tail` options.

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
July 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
April 2015, updated by Ahmet Alp Balkan <ahmetalpbalkan@gmail.com>
