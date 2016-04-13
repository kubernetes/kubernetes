# Examples

## Building
```
$ go get github.com/tools/godep
$ make
```

## Running
### Start Mesos
You will need a running Mesos master and slaves to run the examples.   For instance, start a local Mesos:
```
$ <mesos-build-install>/bin/mesos-local --ip=127.0.0.1 --port=5050 --roles=golang
```
See http://mesos.apache.org/gettingstarted/ for getting started with Apache Mesos.

### Start the Go scheduler/executor examples
```
$ export EXECUTOR_BIN=$(pwd)/_output/executor
$ ./_output/scheduler -master=127.0.0.1:5050 -executor="$EXECUTOR_BIN" -logtostderr=true
```
If all goes well, you should see output about task completion.
You can also point your browser to the Mesos GUI http://127.0.0.1:5050/ to validate the framework activities.

### Start the Go scheduler with other executors
You can also use the Go `example-scheduler` with executors written in other languages such as  `Python` or `Java`  for further validation (note: to use these executors requires a build of the mesos source code with `make check`):
```
$ ./_output/scheduler -master=127.0.0.1:5050 -executor="<mesos-build>/src/examples/python/test-executor" -logtostderr=true
```
Similarly for the Java version:
```
$ ./_output/scheduler -master=127.0.0.1:5050 -executor="<mesos-build>/src/examples/java/test-executor" -logtostderr=true
```

### Start the Go persistent scheduler/executor examples
```
$ export EXECUTOR_BIN=$(pwd)/_output/executor
$ ./_output/persistent_scheduler -master=127.0.0.1:5050 -executor="$EXECUTOR_BIN" -logtostderr=true -role=golang -mesos_authentication_principal=golang
```
