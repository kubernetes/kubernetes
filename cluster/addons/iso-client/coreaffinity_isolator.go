package main

import (
	"flag"

	"github.com/golang/glog"
	"k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity"
	"k8s.io/kubernetes/cluster/addons/iso-client/isolator"
)

const (
	// kubelet eventDispatcher address
	eventDispatcherAddress = "localhost:5433"
	// iso-client own address
	isolatorLocalAddress = "localhost:5444"
	// name of isolator
	name = "cgroup-cpuset-cpus"
)

// TODO: split it to smaller functions
func main() {
	// enable logging to STDERR
	flag.Set("logtostderr", "true")
	flag.Parse()
	glog.Info("Starting ...")

	// creating proper isolator
	coreaffinityIsolator, err := coreaffinity.New(name)
	if err != nil {
		glog.Fatalf("Cannot create coreaffinity isolator: %q", err)
	}

	// Initializing grpc server to handle isolation requests,
	err = isolator.StartIsolatorServer(coreaffinityIsolator, isolatorLocalAddress, eventDispatcherAddress)
	if err != nil {
		glog.Fatalf("Couldn't initialize isolator: %v", err)
	}
}
