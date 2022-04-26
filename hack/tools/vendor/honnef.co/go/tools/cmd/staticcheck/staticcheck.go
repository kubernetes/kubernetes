// staticcheck analyses Go code and makes it better.
package main

import (
	"log"
	"os"

	"honnef.co/go/tools/lintcmd"
	"honnef.co/go/tools/lintcmd/version"
	"honnef.co/go/tools/quickfix"
	"honnef.co/go/tools/simple"
	"honnef.co/go/tools/staticcheck"
	"honnef.co/go/tools/stylecheck"
	"honnef.co/go/tools/unused"
)

func main() {
	cmd := lintcmd.NewCommand("staticcheck")
	cmd.SetVersion(version.Version, version.MachineVersion)

	fs := cmd.FlagSet()
	debug := fs.String("debug.unused-graph", "", "Write unused's object graph to `file`")
	qf := fs.Bool("debug.run-quickfix-analyzers", false, "Run quickfix analyzers")

	cmd.ParseFlags(os.Args[1:])

	cmd.AddAnalyzers(simple.Analyzers...)
	cmd.AddAnalyzers(staticcheck.Analyzers...)
	cmd.AddAnalyzers(stylecheck.Analyzers...)
	cmd.AddAnalyzers(unused.Analyzer)

	if *qf {
		cmd.AddAnalyzers(quickfix.Analyzers...)
	}

	if *debug != "" {
		f, err := os.OpenFile(*debug, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			log.Fatal(err)
		}
		unused.Debug = f
	}

	cmd.Run()
}
