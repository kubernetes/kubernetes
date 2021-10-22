This directory contains auto-generated mocks for the ProfilerService interface.

To regenerate the mocks, install https://github.com/golang/mock tool and run
the command below while in the `mocks` directory.

```
mockgen -package mocks google.golang.org/genproto/googleapis/devtools/cloudprofiler/v2 \
         ProfilerServiceClient \
         > mock_profiler_client.go
```

Then re-add the copyright header in the file. You also need to either run the
commands above using Golang 1.6 or manually change the "context" import to
"context" to ensure the compatibility with Go 1.6.
