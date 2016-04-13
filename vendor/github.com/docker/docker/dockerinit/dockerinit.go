package main

import (
	_ "github.com/docker/docker/daemon/execdriver/lxc"
	_ "github.com/docker/docker/daemon/execdriver/native"
	"github.com/docker/docker/pkg/reexec"
)

func main() {
	// Running in init mode
	reexec.Init()
}
