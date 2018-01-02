package main

import (
	"fmt"
	"io"
	"log"
	"os"

	"github.com/influxdata/influxdb/cmd"
	"github.com/influxdata/influxdb/cmd/influx_inspect/dumptsm"
	"github.com/influxdata/influxdb/cmd/influx_inspect/export"
	"github.com/influxdata/influxdb/cmd/influx_inspect/help"
	"github.com/influxdata/influxdb/cmd/influx_inspect/report"
	"github.com/influxdata/influxdb/cmd/influx_inspect/verify"
	_ "github.com/influxdata/influxdb/tsdb/engine"
)

func main() {

	m := NewMain()
	if err := m.Run(os.Args[1:]...); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// Main represents the program execution.
type Main struct {
	Logger *log.Logger

	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer
}

// NewMain return a new instance of Main.
func NewMain() *Main {
	return &Main{
		Logger: log.New(os.Stderr, "[influx_inspect] ", log.LstdFlags),
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
}

// Run determines and runs the command specified by the CLI args.
func (m *Main) Run(args ...string) error {
	name, args := cmd.ParseCommandName(args)

	// Extract name from args.
	switch name {
	case "", "help":
		if err := help.NewCommand().Run(args...); err != nil {
			return fmt.Errorf("help: %s", err)
		}
	case "dumptsmdev":
		fmt.Fprintf(m.Stderr, "warning: dumptsmdev is deprecated, use dumptsm instead.\n")
		fallthrough
	case "dumptsm":
		name := dumptsm.NewCommand()
		if err := name.Run(args...); err != nil {
			return fmt.Errorf("dumptsm: %s", err)
		}
	case "export":
		name := export.NewCommand()
		if err := name.Run(args...); err != nil {
			return fmt.Errorf("export: %s", err)
		}
	case "report":
		name := report.NewCommand()
		if err := name.Run(args...); err != nil {
			return fmt.Errorf("report: %s", err)
		}
	case "verify":
		name := verify.NewCommand()
		if err := name.Run(args...); err != nil {
			return fmt.Errorf("verify: %s", err)
		}
	default:
		return fmt.Errorf(`unknown command "%s"`+"\n"+`Run 'influx_inspect help' for usage`+"\n\n", name)
	}

	return nil
}
