# InfluxDB Admin Interface

This is the built-in admin interface that ships with InfluxDB. The service is intended to have little overhead and minimal preprocessing steps.

## How it works

Static assets, located in the `assets` directory, are embedded in the `influxd` binary and served from memory using a simple fileserver.

The admin UI itself uses [React](https://github.com/facebook/react) for the user interface to interact directly with the InfluxDB API, usually running on port `8086`.

## Building

The only step required to bundle the admin UI with InfluxDB is to create a compressed file system using `statik` as follows:

```
go get github.com/rakyll/statik  # make sure $GOPATH/bin is listed in your PATH
cd $GOPATH/src/github.com/influxdata/influxdb
go generate github.com/influxdata/influxdb/services/admin
```

The `go generate ./...` command will run `statik` to generate the `statik/statik.go` file. The generated `go` file will embed the admin interface assets into the InfluxDB binary.

This step should be run before submitting any pull requests which include modifications to admin interface assets.
