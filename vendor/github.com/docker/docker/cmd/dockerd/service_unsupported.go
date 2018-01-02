// +build !windows

package main

import (
	"github.com/spf13/pflag"
)

func initService(daemonCli *DaemonCli) (bool, bool, error) {
	return false, false, nil
}

func installServiceFlags(flags *pflag.FlagSet) {
}
