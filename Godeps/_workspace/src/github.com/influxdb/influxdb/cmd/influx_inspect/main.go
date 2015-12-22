package main

import (
	"flag"
	"fmt"
	"os"

	_ "github.com/influxdb/influxdb/tsdb/engine"
)

func usage() {
	println(`Usage: influx_inspect <command> [options]

Displays detailed information about InfluxDB data files.
`)

	println(`Commands:
  info - displays series meta-data for all shards.  Default location [$HOME/.influxdb]
  dumptsm - dumps low-level details about tsm1 files.
  dumptsmdev - dumps low-level details about tsm1dev files.`)
	println()
}

func main() {

	flag.Usage = usage
	flag.Parse()

	if len(flag.Args()) == 0 {
		flag.Usage()
		os.Exit(0)
	}

	switch flag.Args()[0] {
	case "info":
		var path string
		fs := flag.NewFlagSet("info", flag.ExitOnError)
		fs.StringVar(&path, "dir", os.Getenv("HOME")+"/.influxdb", "Root storage path. [$HOME/.influxdb]")

		fs.Usage = func() {
			println("Usage: influx_inspect info [options]\n\n   Displays series meta-data for all shards..")
			println()
			println("Options:")
			fs.PrintDefaults()
		}

		if err := fs.Parse(flag.Args()[1:]); err != nil {
			fmt.Printf("%v", err)
			os.Exit(1)
		}
		cmdInfo(path)
	case "dumptsm":
		var dumpAll bool
		opts := &tsdmDumpOpts{}
		fs := flag.NewFlagSet("file", flag.ExitOnError)
		fs.BoolVar(&opts.dumpIndex, "index", false, "Dump raw index data")
		fs.BoolVar(&opts.dumpBlocks, "blocks", false, "Dump raw block data")
		fs.BoolVar(&dumpAll, "all", false, "Dump all data. Caution: This may print a lot of information")
		fs.StringVar(&opts.filterKey, "filter-key", "", "Only display index and block data match this key substring")

		fs.Usage = func() {
			println("Usage: influx_inspect dumptsm [options] <path>\n\n  Dumps low-level details about tsm1 files.")
			println()
			println("Options:")
			fs.PrintDefaults()
			os.Exit(0)
		}

		if err := fs.Parse(flag.Args()[1:]); err != nil {
			fmt.Printf("%v", err)
			os.Exit(1)
		}

		if len(fs.Args()) == 0 || fs.Args()[0] == "" {
			fmt.Printf("TSM file not specified\n\n")
			fs.Usage()
			fs.PrintDefaults()
			os.Exit(1)
		}
		opts.path = fs.Args()[0]
		opts.dumpBlocks = opts.dumpBlocks || dumpAll || opts.filterKey != ""
		opts.dumpIndex = opts.dumpIndex || dumpAll || opts.filterKey != ""
		cmdDumpTsm1(opts)
	case "dumptsmdev":
		var dumpAll bool
		opts := &tsdmDumpOpts{}
		fs := flag.NewFlagSet("file", flag.ExitOnError)
		fs.BoolVar(&opts.dumpIndex, "index", false, "Dump raw index data")
		fs.BoolVar(&opts.dumpBlocks, "blocks", false, "Dump raw block data")
		fs.BoolVar(&dumpAll, "all", false, "Dump all data. Caution: This may print a lot of information")
		fs.StringVar(&opts.filterKey, "filter-key", "", "Only display index and block data match this key substring")

		fs.Usage = func() {
			println("Usage: influx_inspect dumptsm [options] <path>\n\n  Dumps low-level details about tsm1 files.")
			println()
			println("Options:")
			fs.PrintDefaults()
			os.Exit(0)
		}

		if err := fs.Parse(flag.Args()[1:]); err != nil {
			fmt.Printf("%v", err)
			os.Exit(1)
		}

		if len(fs.Args()) == 0 || fs.Args()[0] == "" {
			fmt.Printf("TSM file not specified\n\n")
			fs.Usage()
			fs.PrintDefaults()
			os.Exit(1)
		}
		opts.path = fs.Args()[0]
		opts.dumpBlocks = opts.dumpBlocks || dumpAll || opts.filterKey != ""
		opts.dumpIndex = opts.dumpIndex || dumpAll || opts.filterKey != ""
		cmdDumpTsm1dev(opts)
	default:
		flag.Usage()
		os.Exit(1)
	}
}
