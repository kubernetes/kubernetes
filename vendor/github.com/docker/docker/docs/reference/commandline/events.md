<!--[metadata]>
+++
title = "events"
description = "The events command description and usage"
keywords = ["events, container, report"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# events

    Usage: docker events [OPTIONS]

    Get real time events from the server

      -f, --filter=[]    Filter output based on conditions provided
      --since=""         Show all events created since timestamp
      --until=""         Stream events until this timestamp

Docker containers will report the following events:

    create, destroy, die, export, kill, oom, pause, restart, start, stop, unpause

and Docker images will report:

    untag, delete

The `--since` and `--until` parameters can be Unix timestamps, RFC3339
dates or Go duration strings (e.g. `10m`, `1h30m`) computed relative to
client machine’s time. If you do not provide the --since option, the command
returns only new and/or live events.

## Filtering

The filtering flag (`-f` or `--filter`) format is of "key=value". If you would
like to use multiple filters, pass multiple flags (e.g., 
`--filter "foo=bar" --filter "bif=baz"`)

Using the same filter multiple times will be handled as a *OR*; for example
`--filter container=588a23dac085 --filter container=a8f7720b8c22` will display
events for container 588a23dac085 *OR* container a8f7720b8c22

Using multiple filters will be handled as a *AND*; for example
`--filter container=588a23dac085 --filter event=start` will display events for
container container 588a23dac085 *AND* the event type is *start*

The currently supported filters are:

* container
* event
* image

## Examples

You'll need two shells for this example.

**Shell 1: Listening for events:**

    $ docker events

**Shell 2: Start and Stop containers:**

    $ docker start 4386fb97867d
    $ docker stop 4386fb97867d
    $ docker stop 7805c1d35632

**Shell 1: (Again .. now showing events):**

    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) start
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) die
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

**Show events in the past from a specified time:**

    $ docker events --since 1378216169
    2014-03-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) die
    2014-03-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

    $ docker events --since '2013-09-03'
    2014-09-03T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) start
    2014-09-03T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) die
    2014-09-03T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

    $ docker events --since '2013-09-03T15:49:29'
    2014-09-03T15:49:29.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) die
    2014-09-03T15:49:29.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

This example outputs all events that were generated in the last 3 minutes,
relative to the current time on the client machine:

    $ docker events --since '3m'
    2015-05-12T11:51:30.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2015-05-12T15:52:12.999999999Z07:00 4 4386fb97867d: (from ubuntu-1:14.04) stop
    2015-05-12T15:53:45.999999999Z07:00  7805c1d35632: (from redis:2.8) die
    2015-05-12T15:54:03.999999999Z07:00  7805c1d35632: (from redis:2.8) stop

**Filter events:**

    $ docker events --filter 'event=stop'
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop
    2014-09-03T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

    $ docker events --filter 'image=ubuntu-1:14.04'
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) start
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop

    $ docker events --filter 'container=7805c1d35632'
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) die
    2014-09-03T15:49:29.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

    $ docker events --filter 'container=7805c1d35632' --filter 'container=4386fb97867d'
    2014-09-03T15:49:29.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) die
    2014-09-03T15:49:29.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

    $ docker events --filter 'container=7805c1d35632' --filter 'event=stop'
    2014-09-03T15:49:29.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

    $ docker events --filter 'container=container_1' --filter 'container=container_2'
    2014-09-03T15:49:29.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) die
    2014-05-10T17:42:14.999999999Z07:00 4386fb97867d: (from ubuntu-1:14.04) stop
    2014-05-10T17:42:14.999999999Z07:00 7805c1d35632: (from redis:2.8) die
    2014-09-03T15:49:29.999999999Z07:00 7805c1d35632: (from redis:2.8) stop

