# Integration Testing cAdvisor

The cAdvisor integration tests can be found in `integration/tests`. These run queries on a running cAdvisor. To run these tests:

```
$ go run integration/runner/runner.go -port=PORT <hosts to test>
```

This will build a cAdvisor from the current repository and start it on the target machine before running the tests.

To simply run the tests against an existing cAdvisor:

```
$ go test github.com/google/cadvisor/integration/tests/... -host=HOST -port=PORT
```

Note that `HOST` and `PORT` default to `localhost` and `8080` respectively.
Today We only support remote execution in Google Compute Engine since that is where we run our continuous builds.
