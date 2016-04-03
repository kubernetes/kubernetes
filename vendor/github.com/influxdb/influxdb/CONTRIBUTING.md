Contributing to InfluxDB
========================

Bug reports
---------------
Before you file an issue, please search existing issues in case it has already been filed, or perhaps even fixed. If you file an issue, please include the following.
* Full details of your operating system (or distribution) e.g. 64-bit Ubuntu 14.04.
* The version of InfluxDB you are running
* Whether you installed it using a pre-built package, or built it from source.
* A small test case, if applicable, that demonstrates the issues.

Remember the golden rule of bug reports: **The easier you make it for us to reproduce the problem, the faster it will get fixed.**
If you have never written a bug report before, or if you want to brush up on your bug reporting skills, we recommend reading [Simon Tatham's essay "How to Report Bugs Effectively."](http://www.chiark.greenend.org.uk/~sgtatham/bugs.html)

Test cases should be in the form of `curl` commands. For example:
```
# create database
curl -G http://localhost:8086/query --data-urlencode "q=CREATE DATABASE mydb"

# create retention policy
curl -G http://localhost:8086/query --data-urlencode "q=CREATE RETENTION POLICY myrp ON mydb DURATION 365d REPLICATION 1 DEFAULT"

# write data
curl -d '{"database" : "mydb", "retentionPolicy" : "myrp", "points": [{"measurement":"cpu","tags":{"region":"useast","host":"server_1","service":"redis"},"fields":{"value":61}}]}' -H "Content-Type: application/json" http://localhost:8086/write

# Delete a Measurement
curl -G http://localhost:8086/query  --data-urlencode 'db=mydb' --data-urlencode 'q=DROP MEASUREMENT cpu'

# Query the Measurement, it should return no data, but data comes back.
curl -G http://localhost:8086/query  --data-urlencode 'db=mydb' --data-urlencode 'SELECT * from cpu'
```
**If you don't include a clear test case like this, your issue may not be investigated, and may even be closed**. If writing the data is too difficult, please zip up your data directory and include a link to it in your bug report.

Please note that issues are *not the place to file general questions* such as "how do I use collectd with InfluxDB?" Questions of this nature should be sent to the [Google Group](https://groups.google.com/forum/#!forum/influxdb), not filed as issues. Issues like this will be closed.

Feature requests
---------------
We really like to receive feature requests, as it helps us prioritize our work. Please be clear about your requirements, as incomplete feature requests may simply be closed if we don't understand what you would like to see added to InfluxDB.

Contributing to the source code
---------------

InfluxDB follows standard Go project structure. This means that all
your go development are done in `$GOPATH/src`. GOPATH can be any
directory under which InfluxDB and all its dependencies will be
cloned. For more details on recommended go project's structure, see
[How to Write Go Code](http://golang.org/doc/code.html) and
[Go: Best Practices for Production Environments](http://peter.bourgon.org/go-in-production/), or you can just follow
the steps below.

Submitting a pull request
------------
To submit a pull request you should fork the InfluxDB repository, and make your change on a feature branch of your fork. Then generate a pull request from your branch against *master* of the InfluxDB repository. Include in your pull request details of your change -- the why *and* the how -- as well as the testing your performed. Also, be sure to run the test suite with your change in place. Changes that cause tests to fail cannot be merged.

There will usually be some back and forth as we finalize the change, but once that completes it may be merged.

To assist in review for the PR, please add the following to your pull request comment:

```md
- [ ] CHANGELOG.md updated
- [ ] Rebased/mergable
- [ ] Tests pass
- [ ] Sign [CLA](http://influxdb.com/community/cla.html) (if not already signed)
```

Use of third-party packages
------------
A third-party package is defined as one that is not part of the standard Go distribution. Generally speaking we prefer to minimize our use of third-party packages, and avoid them unless absolutely necessarly. We'll often write a little bit of code rather than pull in a third-party package. Of course, we do use some third-party packages -- most importantly we use [BoltDB](https://github.com/boltdb/bolt) as the storage engine. So to maximise the chance your change will be accepted by us, use only the standard libaries, or the third-party packages we have decided to use.

For rationale, check out the post [The Case Against Third Party Libraries](http://blog.gopheracademy.com/advent-2014/case-against-3pl/).

Signing the CLA
---------------

If you are going to be contributing back to InfluxDB please take a
second to sign our CLA, which can be found
[on our website](http://influxdb.com/community/cla.html).

Installing Go
-------------
InfluxDB requires Go 1.4 or greater.

At InfluxDB we find gvm, a Go version manager, useful for installing Go. For instructions
on how to install it see [the gvm page on github](https://github.com/moovweb/gvm).

After installing gvm you can install and set the default go version by
running the following:

    gvm install go1.4
    gvm use go1.4 --default

Revision Control Systems
------
Go has the ability to import remote packages via revision control systems with the `go get` command.  To ensure that you can retrieve any remote package, be sure to install the following rcs software to your system.
Currently the project only depends on `git` and `mercurial`.

* [Install Git](http://git-scm.com/book/en/Getting-Started-Installing-Git)
* [Install Mercurial](http://mercurial.selenic.com/wiki/Download)

Project structure
-----------------
First you need to setup the project structure:

    export GOPATH=$HOME/gocodez
    mkdir -p $GOPATH/src/github.com/influxdb
    cd $GOPATH/src/github.com/influxdb
    git clone git@github.com:influxdb/influxdb

You can add the line `export GOPATH=$HOME/gocodez` to your bash/zsh
file to be set for every shell instead of having to manually run it
everytime.

We have a pre commit hook to make sure code is formatted properly
and vetted before you commit any changes. We strongly recommend using the pre
commit hook to guard against accidentally committing unformatted
code. To use the pre-commit hook, run the following:

    cd $GOPATH/src/github.com/influxdb/influxdb
    cp .hooks/pre-commit .git/hooks/

In case the commit is rejected because it's not formatted you can run
the following to format the code:

```
go fmt ./...
go vet ./...
```

To install go vet, run the following command:
```
go get golang.org/x/tools/cmd/vet
```

NOTE: If you have not installed mercurial, the above command will fail.  See [Revision Control Systems](#revision-control-systems) above.

For more information on `go vet`, [read the GoDoc](https://godoc.org/golang.org/x/tools/cmd/vet).

Build and Test
-----

Make sure you have Go installed and the project structure as shown above. To then build the project, execute the following commands:

```bash
cd $GOPATH/src/github.com/influxdb
go get -u -f -t ./...
go build ./...
```

To then install the binaries, run the following command. They can be found in `$GOPATH/bin`. Please note that the InfluxDB binary is named `influxd`, not `influxdb`.

```bash
go install ./...
```

To set the version and commit flags during the build pass the following to the build command:

```bash
-ldflags="-X main.version $VERSION -X main.commit $COMMIT"
```

where `$VERSION` is the version, and `$COMMIT` is the git commit hash.

To run the tests, execute the following command:

```bash
cd $GOPATH/src/github.com/influxdb/influxdb
go test -v ./...

# run tests that match some pattern
go test -run=TestDatabase . -v

# run tests and show coverage
go test -coverprofile /tmp/cover . && go tool cover -html /tmp/cover
```

To install go cover, run the following command:
```
go get golang.org/x/tools/cmd/cover
```

Generated Google Protobuf code
-----------------
Most changes to the source do not require that the generated protocol buffer code be changed. But if you need to modify the protocol buffer code, you'll first need to install the protocol buffers toolchain.

First install the [protocol buffer compiler](https://developers.google.com/protocol-buffers/
) 2.6.1 or later for your OS:

Then install the go plugins:

```bash
go get github.com/gogo/protobuf/proto
go get github.com/gogo/protobuf/protoc-gen-gogo
go get github.com/gogo/protobuf/gogoproto
```

Finally run, `go generate` after updating any `*.proto` file:

```bash
go generate ./...
```
**Trouleshooting**

If generating the protobuf code is failing for you, check each of the following:
 * Ensure the protobuf library can be found. Make sure that `LD_LIBRRARY_PATH` includes the directory in which the library `libprotoc.so` has been installed.
 * Ensure the command `protoc-gen-gogo`, found in `GOPATH/bin`, is on your path. This can be done by adding `GOPATH/bin` to `PATH`.

Continuous Integration testing
-----
InfluxDB uses CirceCI for continuous integration testing. To see how the code is built and tested, check out [this file](https://github.com/influxdb/influxdb/blob/master/circle-test.sh). It closely follows the build and test process outlined above. You can see the exact version of Go InfluxDB uses for testing by consulting that file.

Useful links
------------
- [Useful techniques in Go](http://arslan.io/ten-useful-techniques-in-go)
- [Go in production](http://peter.bourgon.org/go-in-production/)
- [Principles of designing Go APIs with channels](https://inconshreveable.com/07-08-2014/principles-of-designing-go-apis-with-channels/)
- [Common mistakes in Golang](http://soryy.com/blog/2014/common-mistakes-with-go-lang/). Especially this section `Loops, Closures, and Local Variables`
