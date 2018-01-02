// +build !windows

package main

import "github.com/containerd/containerd/cmd/ctr/commands/shim"

func init() {
	extraCmds = append(extraCmds, shim.Command)
}
