Siege
=====

Siege is an HTTP benchmarking tool and can be used against InfluxDB easily.
If you're on Mac you can install `siege` using `brew`. If you're on Linux
you can install using your package manager.


## Initializing the database

Before you run your siege, you need to do 2 things:

- Create a database named `db`.
- Create a retention policy named `raw`.

You can do this with the following commands:

```sh
$ curl -G http://localhost:8086/query --data-urlencode "q=CREATE DATABASE db"
$ curl -G http://localhost:8086/query --data-urlencode "q=CREATE RETENTION POLICY raw ON db DURATION 30d REPLICATION 3 DEFAULT"
```


## Running

To run siege, first start one or more InfluxDB nodes. At least one of those
nodes should run on the default port of `8086`.

Next, generate a URL file to run. You can use the `urlgen` utility in this
folder to make the file. Simply set the number of unique clients and number of
series to generate:

```sh
$ ./urlgen -c 10 -s 100 > urls.txt
```

Now you can execute siege. There are several arguments available but only 
a few that we're concerned with:

```
-c NUM   the number of concurrent connections.
-d NUM   delay between each request, in seconds.
-b       benchmark mode. runs with a delay of 0.
-t DUR   duration of the benchmark. value should end in 's', 'm', or 'h'.
-f FILE  the path to the URL file.
```

These can be combined to simulate different load. For example, this command
will execute writes against using 100 concurrent connections with a 1 second
delay in between each call:

```sh
$ siege -c 100 -f urls.txt
```

Again, you can also specify the `-b` option to remove the delay.


## Verification

You can verify that your data made it in by executing a query against it:

```sh
$ curl -G http://localhost:8086/query --data-urlencode "db=db" --data-urlencode "q=SELECT sum(value) FROM cpu GROUP BY time(1h)"
```

