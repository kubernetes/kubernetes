<!--[metadata]>
+++
title = "logs"
description = "The logs command description and usage"
keywords = ["logs, retrieve, docker"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# logs

    Usage: docker logs [OPTIONS] CONTAINER

    Fetch the logs of a container

      -f, --follow=false        Follow log output
      --since=""                Show logs since timestamp
      -t, --timestamps=false    Show timestamps
      --tail="all"              Number of lines to show from the end of the logs

NOTE: this command is available only for containers with `json-file` logging
driver.

The `docker logs` command batch-retrieves logs present at the time of execution.

The `docker logs --follow` command will continue streaming the new output from
the container's `STDOUT` and `STDERR`.

Passing a negative number or a non-integer to `--tail` is invalid and the
value is set to `latest` in that case.

The `docker logs --timestamp` commands will add an RFC3339Nano
timestamp, for example `2014-09-16T06:17:46.000000000Z`, to each
log entry. To ensure that the timestamps for are aligned the
nano-second part of the timestamp will be padded with zero when necessary.

The `--since` option shows only the container logs generated after
a given date. You can specify the date as an RFC 3339 date, a UNIX
timestamp, or a Go duration string (e.g. `1m30s`, `3h`). Docker computes
the date relative to the client machine’s time. You can combine
the `--since` option with either or both of the `--follow` or `--tail` options.
