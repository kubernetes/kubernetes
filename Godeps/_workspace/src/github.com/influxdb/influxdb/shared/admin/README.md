# InfluxDB Admin Interface

This is the built-in admin interface that ships with InfluxDB. An emphasis has been placed on making it minimalistic and able to be bundled with the server without adding lots of overhead or preprocessing steps.

## Building

The only step required to bundle this with InfluxDB is to create a compressed file system using `statik` as follows:

```
go get github.com/rakyll/statik
$GOPATH/bin/statik -src=./shared/admin
go fmt statik/statik.go
```

That should update the file located at `statik/statik.go`, which will get automatically build into the InfluxDB binary when it is compiled. This step should also be completed in any pull requests that make modifications to the admin interface, as it won't be run as a separate step in the release process.
