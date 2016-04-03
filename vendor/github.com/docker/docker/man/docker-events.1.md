% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-events - Get real time events from the server

# SYNOPSIS
**docker events**
[**--help**]
[**-f**|**--filter**[=*[]*]]
[**--since**[=*SINCE*]]
[**--until**[=*UNTIL*]]


# DESCRIPTION
Get event information from the Docker daemon. Information can include historical
information and real-time information.

Docker containers will report the following events:

    create, destroy, die, export, kill, pause, restart, start, stop, unpause

and Docker images will report:

    untag, delete

# OPTIONS
**--help**
  Print usage statement

**-f**, **--filter**=[]
   Provide filter values (i.e., 'event=stop')

**--since**=""
   Show all events created since timestamp

**--until**=""
   Stream events until this timestamp

You can specify `--since` and `--until` parameters as an RFC 3339 date,
a UNIX timestamp, or a Go duration string (e.g. `1m30s`, `3h`). Docker computes
the date relative to the client machine’s time.

# EXAMPLES

## Listening for Docker events

After running docker events a container 786d698004576 is started and stopped
(The container name has been shortened in the output below):

    # docker events
    2015-01-28T20:21:31.000000000-08:00 59211849bc10: (from whenry/testimage:latest) start
    2015-01-28T20:21:31.000000000-08:00 59211849bc10: (from whenry/testimage:latest) die
    2015-01-28T20:21:32.000000000-08:00 59211849bc10: (from whenry/testimage:latest) stop

## Listening for events since a given date
Again the output container IDs have been shortened for the purposes of this document:

    # docker events --since '2015-01-28'
    2015-01-28T20:25:38.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) create
    2015-01-28T20:25:38.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) start
    2015-01-28T20:25:39.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) create
    2015-01-28T20:25:39.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) start
    2015-01-28T20:25:40.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) die
    2015-01-28T20:25:42.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) stop
    2015-01-28T20:25:45.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) start
    2015-01-28T20:25:45.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) die
    2015-01-28T20:25:46.000000000-08:00 c21f6c22ba27: (from whenry/testimage:latest) stop

The following example outputs all events that were generated in the last 3 minutes,
relative to the current time on the client machine:

    # docker events --since '3m'
    2015-05-12T11:51:30.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2015-05-12T15:52:12.999999999Z07:00 4 4386fb97867d: (from ubuntu-1:14.04) stop
    2015-05-12T15:53:45.999999999Z07:00  7805c1d35632: (from redis:2.8) die
    2015-05-12T15:54:03.999999999Z07:00  7805c1d35632: (from redis:2.8) stop

If you do not provide the --since option, the command returns only new and/or
live events.

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
June 2015, updated by Brian Goff <cpuguy83@gmail.com>
