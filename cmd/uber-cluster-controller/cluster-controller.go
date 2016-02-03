package main

import (
	"os"
	"fmt"
	"k8s.io/kubernetes/cmd/uber-cluster-controller/app"
	"k8s.io/kubernetes/cmd/uber-cluster-controller/app/options"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/version/verflag"
	"github.com/spf13/pflag"
)


func main() {
	c := options.NewClusterController()
	c.AddFlags(pflag.CommandLine)

	util.InitFlags()
	util.InitLogs()
	defer util.FlushLogs()

	verflag.PrintAndExitIfRequested()

	if err := app.Run(c); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
