package main

import (
	"flag"

	"github.com/golang/glog"
	"k8s.io/kubernetes/cluster/addons/iso-client/isolator"
	"k8s.io/kubernetes/cluster/addons/iso-client/noop"
)

const (
	// kubelet eventDispatcher address
	remoteAddress = "localhost:5433"
	// iso-client own address
	localAddress = "localhost:5444"
	// name of isolator
	name = "noop"
)

func main() {
	// enable logging to STDERR
	flag.Set("logtostderr", "true")
	flag.Parse()
	glog.Info("Starting ...")

	// Create isolator
	iso, err := noop.New(name)
	if err != nil {
		glog.Fatalf("Cannot create isolator: %q", err)
	}

	// Start grpc server to handle isolation events
	err = isolator.StartIsolatorServer(iso, localAddress, remoteAddress)
	if err != nil {
		glog.Fatalf("Couldn't initialize isolator: %v", err)
	}
}
