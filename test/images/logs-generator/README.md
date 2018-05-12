# Logs Generator

## Overview

Logs generator is a tool to create predictable load on the logs delivery system.
Is generates random lines with predictable format and predictable average length.
Each line can be later uniquely identified to ensure logs delivery.

## Usage

Tool is parametrized with the total number of number that should be generated and the duration of
the generation process. For example, if you want to create a throughput of 100 lines per second
for a minute, you set total number of lines to 6000 and duration to 1 minute.

Parameters are passed through environment variables. There are no defaults, you should always 
set up container parameters. Total number of line is parametrized through env variable
`LOGS_GENERATOR_LINES_TOTAL` and duration in go format is parametrized through env variable
`LOGS_GENERATOR_DURATION`.

Inside the container all log lines are written to the stdout.

Each line is on average 100 bytes long and follows this pattern:

```
2000-12-31T12:59:59Z <id> <method> /api/v1/namespaces/<namespace>/endpoints/<random_string> <random_number>
```

Where `<id>` refers to the number from 0 to `total_lines - 1`, which is unique for each
line in a given run of the container.

## Image

Image is located in the public repository of Google Container Registry under the name

```
k8s.gcr.io/logs-generator:v0.1.1
```

## Examples

```
docker run -i \
  -e "LOGS_GENERATOR_LINES_TOTAL=10" \
  -e "LOGS_GENERATOR_DURATION=1s" \
  k8s.gcr.io/logs-generator:v0.1.1
```

```
kubectl run logs-generator \
  --generator=run-pod/v1 \
  --image=k8s.gcr.io/logs-generator:v0.1.1 \
  --restart=Never \
  --env "LOGS_GENERATOR_LINES_TOTAL=1000" \
  --env "LOGS_GENERATOR_DURATION=1m"
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/images/logs-generator/README.md?pixel)]()
