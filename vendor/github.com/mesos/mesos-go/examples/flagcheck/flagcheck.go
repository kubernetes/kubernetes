package main

import (
	"flag"
	"fmt"
	"os"

	// Import all non-testing packages to verify that flags are not added
	// to the command line.

	_ "github.com/mesos/mesos-go/auth"
	_ "github.com/mesos/mesos-go/auth/callback"
	_ "github.com/mesos/mesos-go/auth/sasl"
	_ "github.com/mesos/mesos-go/auth/sasl/mech"
	_ "github.com/mesos/mesos-go/auth/sasl/mech/crammd5"
	_ "github.com/mesos/mesos-go/detector"
	_ "github.com/mesos/mesos-go/detector/zoo"
	_ "github.com/mesos/mesos-go/executor"
	_ "github.com/mesos/mesos-go/healthchecker"
	_ "github.com/mesos/mesos-go/mesos"
	_ "github.com/mesos/mesos-go/mesosproto"
	_ "github.com/mesos/mesos-go/mesosproto/scheduler"
	_ "github.com/mesos/mesos-go/mesosutil"
	_ "github.com/mesos/mesos-go/mesosutil/process"
	_ "github.com/mesos/mesos-go/messenger"
	_ "github.com/mesos/mesos-go/messenger/sessionid"
	_ "github.com/mesos/mesos-go/scheduler"
	_ "github.com/mesos/mesos-go/upid"
)

// Flags which are accepted from other packages.
var allowedFlags = []string{
	// Flags added from the glog package
	"logtostderr",
	"alsologtostderr",
	"v",
	"stderrthreshold",
	"vmodule",
	"log_backtrace_at",
	"log_dir",
}

func main() {
	expected := map[string]struct{}{}
	for _, f := range allowedFlags {
		expected[f] = struct{}{}
	}

	hasLeak := false
	flag.CommandLine.VisitAll(func(f *flag.Flag) {
		if _, ok := expected[f.Name]; !ok {
			fmt.Fprintf(os.Stderr, "Leaking flag %q: %q\n", f.Name, f.Usage)
			hasLeak = true
		}
	})

	if hasLeak {
		os.Exit(1)
	}
}
