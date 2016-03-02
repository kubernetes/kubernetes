# InfluxDB Client

[![GoDoc](https://godoc.org/github.com/influxdb/influxdb?status.svg)](http://godoc.org/github.com/influxdb/influxdb/client)

## Description

A Go client library written and maintained by the **InfluxDB** team.
This package provides convenience functions to read and write time series data.
It uses the HTTP protocol to communicate with your **InfluxDB** cluster.


## Getting Started

### Connecting To Your Database

Connecting to an **InfluxDB** database is straightforward. You will need a host
name, a port and the cluster user credentials if applicable. The default port is 8086.
You can customize these settings to your specific installation via the
**InfluxDB** configuration file.

Thought not necessary for experimentation, you may want to create a new user
and authenticate the connection to your database.

For more information please check out the
[Cluster Admin Docs](http://influxdb.com/docs/v0.9/query_language/database_administration.html).

For the impatient, you can create a new admin user _bubba_ by firing off the
[InfluxDB CLI](https://github.com/influxdb/influxdb/blob/master/cmd/influx/main.go).

```shell
influx
> create user bubba with password 'bumblebeetuna'
> grant all privileges to bubba
```

And now for good measure set the credentials in you shell environment.
In the example below we will use $INFLUX_USER and $INFLUX_PWD

Now with the administrivia out of the way, let's connect to our database.

NOTE: If you've opted out of creating a user, you can omit Username and Password in
the configuration below.

```go
package main

import "github.com/influxdb/influxdb/client"

const (
	MyHost        = "localhost"
	MyPort        = 8086
	MyDB          = "square_holes"
	MyMeasurement = "shapes"
)

func main() {
	u, err := url.Parse(fmt.Sprintf("http://%s:%d", MyHost, MyPort))
	if err != nil {
		log.Fatal(err)
	}

	conf := client.Config{
		URL:      *u,
		Username: os.Getenv("INFLUX_USER"),
		Password: os.Getenv("INFLUX_PWD"),
	}

	con, err := client.NewClient(conf)
	if err != nil {
		log.Fatal(err)
	}

	dur, ver, err := con.Ping()
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Happy as a Hippo! %v, %s", dur, ver)
}

```

### Inserting Data

Time series data aka *points* are written to the database using batch inserts.
The mechanism is to create one or more points and then create a batch aka *batch points*
and write these to a given database and series. A series is a combination of a
measurement (time/values) and a set of tags.

In this sample we will create a batch of a 1,000 points. Each point has a time and
a single value as well as 2 tags indicating a shape and color. We write these points
to a database called _square_holes_ using a measurement named _shapes_.

NOTE: You can specify a RetentionPolicy as part of the batch points. If not
provided InfluxDB will use the database _default_ retention policy. By default, the _default_
retention policy never deletes any data it contains.

```go
func writePoints(con *client.Client) {
	var (
		shapes     = []string{"circle", "rectangle", "square", "triangle"}
		colors     = []string{"red", "blue", "green"}
		sampleSize = 1000
		pts        = make([]client.Point, sampleSize)
	)

	rand.Seed(42)
	for i := 0; i < sampleSize; i++ {
		pts[i] = client.Point{
			Measurement: "shapes",
			Tags: map[string]string{
				"color": strconv.Itoa(rand.Intn(len(colors))),
				"shape": strconv.Itoa(rand.Intn(len(shapes))),
			},
			Fields: map[string]interface{}{
				"value": rand.Intn(sampleSize),
			},
			Time: time.Now(),
			Precision: "s",
		}
	}

	bps := client.BatchPoints{
		Points:          pts,
		Database:        MyDB,
		RetentionPolicy: "default",
	}
	_, err := con.Write(bps)
	if err != nil {
		log.Fatal(err)
	}
}
```


### Querying Data

One nice advantage of using **InfluxDB** the ability to query your data using familiar
SQL constructs. In this example we can create a convenience function to query the database
as follows:

```go
// queryDB convenience function to query the database
func queryDB(con *client.Client, cmd string) (res []client.Result, err error) {
	q := client.Query{
		Command:  cmd,
		Database: MyDB,
	}
	if response, err := con.Query(q); err == nil {
		if response.Error() != nil {
			return res, response.Error()
		}
		res = response.Results
	}
	return
}
```

#### Creating a Database
```go
_, err := queryDB(con, fmt.Sprintf("create database %s", MyDB))
if err != nil {
	log.Fatal(err)
}
```

#### Count Records
```go
q := fmt.Sprintf("select count(%s) from %s", "value", MyMeasurement)
res, err := queryDB(con, q)
if err != nil {
	log.Fatal(err)
}
count := res[0].Series[0].Values[0][1]
log.Printf("Found a total of `%v records", count)

```

#### Find the last 10 _shapes_ records

```go
q := fmt.Sprintf("select * from %s limit %d", MyMeasurement, 20)
res, err = queryDB(con, q)
if err != nil {
	log.Fatal(err)
}

for i, row := range res[0].Series[0].Values {
	t, err := time.Parse(time.RFC3339, row[0].(string))
	if err != nil {
		log.Fatal(err)
	}
	val, err := row[1].(json.Number).Int64()
	log.Printf("[%2d] %s: %03d\n", i, t.Format(time.Stamp), val)
}
```

## Go Docs

Please refer to
[http://godoc.org/github.com/influxdb/influxdb/client](http://godoc.org/github.com/influxdb/influxdb/client)
for documentation.

## See Also

You can also examine how the client library is used by the
[InfluxDB CLI](https://github.com/influxdb/influxdb/blob/master/cmd/influx/main.go).
