# Loader

This directory includes loader application to allow throughput analysis of Kubernetes logging. 

## Load testing



Run:
```console
go run ./staging/src/k8s.io/component-base/logs/loader/cmd/logger.go 1>/tmp/stdout.out 2>/tmp/stderr.out && du /tmp/*.out
```

Expected output:
```
0       /tmp/stderr.out
340232  /tmp/stdout.out
```

To calculate throughput add size of both files and divide by run duration (default 10s) to get KiB/s.
From the example above we would get `(0 + 340232) / 10 = 34023.2 KiB/s = 33.2 MiB/s`.

Configuration matrix:
* text format - `--logging-format=text`
* json format - `--logging-format=json`
* json with split streams - `--logging-format=json --log-json-split-stream=true`
* json with info buffers (sizes from `1Ki` to `100Mi`) - `--logging-format=json --log-json-split-stream=true --log-json-info-buffer-size=1Ki`
