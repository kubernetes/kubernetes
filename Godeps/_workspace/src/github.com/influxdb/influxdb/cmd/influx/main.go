package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/influxdb/influxdb/client"
	"github.com/influxdb/influxdb/cmd/influx/cli"
)

// These variables are populated via the Go linker.
var (
	version = "0.9"
)

const (
	// defaultFormat is the default format of the results when issuing queries
	defaultFormat = "column"

	// defaultPrecision is the default timestamp format of the results when issuing queries
	defaultPrecision = "ns"

	// defaultPPS is the default points per second that the import will throttle at
	// by default it's 0, which means it will not throttle
	defaultPPS = 0
)

func main() {
	c := cli.New(version)

	fs := flag.NewFlagSet("InfluxDB shell version "+version, flag.ExitOnError)
	fs.StringVar(&c.Host, "host", client.DefaultHost, "Influxdb host to connect to.")
	fs.IntVar(&c.Port, "port", client.DefaultPort, "Influxdb port to connect to.")
	fs.StringVar(&c.Username, "username", c.Username, "Username to connect to the server.")
	fs.StringVar(&c.Password, "password", c.Password, `Password to connect to the server.  Leaving blank will prompt for password (--password="").`)
	fs.StringVar(&c.Database, "database", c.Database, "Database to connect to the server.")
	fs.BoolVar(&c.Ssl, "ssl", false, "Use https for connecting to cluster.")
	fs.StringVar(&c.Format, "format", defaultFormat, "Format specifies the format of the server responses:  json, csv, or column.")
	fs.StringVar(&c.Precision, "precision", defaultPrecision, "Precision specifies the format of the timestamp:  rfc3339,h,m,s,ms,u or ns.")
	fs.StringVar(&c.WriteConsistency, "consistency", "any", "Set write consistency level: any, one, quorum, or all.")
	fs.BoolVar(&c.Pretty, "pretty", false, "Turns on pretty print for the json format.")
	fs.StringVar(&c.Execute, "execute", c.Execute, "Execute command and quit.")
	fs.BoolVar(&c.ShowVersion, "version", false, "Displays the InfluxDB version.")
	fs.BoolVar(&c.Import, "import", false, "Import a previous database.")
	fs.IntVar(&c.PPS, "pps", defaultPPS, "How many points per second the import will allow.  By default it is zero and will not throttle importing.")
	fs.StringVar(&c.Path, "path", "", "path to the file to import")
	fs.BoolVar(&c.Compressed, "compressed", false, "set to true if the import file is compressed")

	// Define our own custom usage to print
	fs.Usage = func() {
		fmt.Println(`Usage of influx:
  -version
       Display the version and exit.
  -host 'host name'
       Host to connect to.
  -port 'port #'
       Port to connect to.
  -database 'database name'
       Database to connect to the server.
  -password 'password'
      Password to connect to the server.  Leaving blank will prompt for password (--password '').
  -username 'username'
       Username to connect to the server.
  -ssl
        Use https for requests.
  -execute 'command'
       Execute command and quit.
  -format 'json|csv|column'
       Format specifies the format of the server responses:  json, csv, or column.
  -precision 'rfc3339|h|m|s|ms|u|ns'
       Precision specifies the format of the timestamp:  rfc3339, h, m, s, ms, u or ns.
  -consistency 'any|one|quorum|all'
       Set write consistency level: any, one, quorum, or all
  -pretty
       Turns on pretty print for the json format.
  -import
       Import a previous database export from file
  -pps
       How many points per second the import will allow.  By default it is zero and will not throttle importing.
  -path
       Path to file to import
  -compressed
       Set to true if the import file is compressed

Examples:

    # Use influx in a non-interactive mode to query the database "metrics" and pretty print json:
    $ influx -database 'metrics' -execute 'select * from cpu' -format 'json' -pretty

    # Connect to a specific database on startup and set database context:
    $ influx -database 'metrics' -host 'localhost' -port '8086'
`)
	}
	fs.Parse(os.Args[1:])

	if c.ShowVersion {
		c.Version()
		os.Exit(0)
	}

	c.Run()
}
