<!--[metadata]>
+++
title = "journald logging driver"
description = "Describes how to use the fluentd logging driver."
keywords = ["Fluentd, docker, logging, driver"]
[menu.main]
parent = "smn_logging"
+++
<![end-metadata]-->

# Journald logging driver

The `journald` logging driver sends container logs to the [systemd
journal](http://www.freedesktop.org/software/systemd/man/systemd-journald.service.html).  Log entries can be retrieved using the `journalctl`
command or through use of the journal API.

In addition to the text of the log message itself, the `journald` log
driver stores the following metadata in the journal with each message:

| Field               | Description |
----------------------|-------------|
| `CONTAINER_ID`      | The container ID truncated to 12 characters. |
| `CONTAINER_ID_FULL` | The full 64-character container ID. |
| `CONTAINER_NAME`    | The container name at the time it was started. If you use `docker rename` to rename a container, the new name is not reflected in the journal entries. |

## Usage

You can configure the default logging driver by passing the
`--log-driver` option to the Docker daemon:

    docker --log-driver=journald

You can set the logging driver for a specific container by using the
`--log-driver` option to `docker run`:

    docker run --log-driver=journald ...

## Note regarding container names

The value logged in the `CONTAINER_NAME` field is the container name
that was set at startup.  If you use `docker rename` to rename a
container, the new name will not be reflected in the journal entries.
Journal entries will continue to use the original name.

## Retrieving log messages with journalctl

You can use the `journalctl` command to retrieve log messages.  You
can apply filter expressions to limit the retrieved messages to a
specific container.  For example, to retrieve all log messages from a
container referenced by name:

    # journalctl CONTAINER_NAME=webserver

You can make use of additional filters to further limit the messages
retrieved.  For example, to see just those messages generated since
the system last booted:

    # journalctl -b CONTAINER_NAME=webserver

Or to retrieve log messages in JSON format with complete metadata:

    # journalctl -o json CONTAINER_NAME=webserver

## Retrieving log messages with the journal API

This example uses the `systemd` Python module to retrieve container
logs:

    import systemd.journal

    reader = systemd.journal.Reader()
    reader.add_match('CONTAINER_NAME=web')

    for msg in reader:
      print '{CONTAINER_ID_FULL}: {MESSAGE}'.format(**msg)

