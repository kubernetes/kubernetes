# Example

This directory includes example logger setup allowing users to easily check and test impact of logging configuration. 

Below we can see examples of how some features work.

## Default

Run:
```console
go run ./staging/src/k8s.io/component-base/logs/example/cmd/logger.go
```

Expected output:
```
I0605 22:03:07.224293 3228948 logger.go:58] Log using Infof, key: value
I0605 22:03:07.224378 3228948 logger.go:59] "Log using InfoS" key="value"
E0605 22:03:07.224393 3228948 logger.go:61] Log using Errorf, err: fail
E0605 22:03:07.224402 3228948 logger.go:62] "Log using ErrorS" err="fail"
I0605 22:03:07.224407 3228948 logger.go:64] Log message has been redacted. Log argument #0 contains: [secret-key]
```

## JSON 

Run:
```console
go run ./staging/src/k8s.io/component-base/logs/example/cmd/logger.go --logging-format json
```

Expected output:
```
{"ts":1624215726270.3562,"caller":"cmd/logger.go:58","msg":"Log using Infof, key: value\n","v":0}
{"ts":1624215726270.4377,"caller":"cmd/logger.go:59","msg":"Log using InfoS","v":0,"key":"value"}
{"ts":1624215726270.6724,"caller":"cmd/logger.go:61","msg":"Log using Errorf, err: fail\n","v":0}
{"ts":1624215726270.7566,"caller":"cmd/logger.go:62","msg":"Log using ErrorS","err":"fail","v":0}
{"ts":1624215726270.8428,"caller":"cmd/logger.go:64","msg":"Log with sensitive key, data: {\"secret\"}\n","v":0}
```

## Logging sanitization

```console
go run ./staging/src/k8s.io/component-base/logs/example/cmd/logger.go --experimental-logging-sanitization
```

Expected output:
```
I0605 22:04:02.019609 3229645 logger.go:58] Log using Infof, key: value
I0605 22:04:02.019677 3229645 logger.go:59] "Log using InfoS" key="value"
E0605 22:04:02.019698 3229645 logger.go:61] Log using Errorf, err: fail
E0605 22:04:02.019709 3229645 logger.go:62] "Log using ErrorS" err="fail"
I0605 22:04:02.019714 3229645 logger.go:64] Log message has been redacted. Log argument #0 contains: [secret-key]
```
