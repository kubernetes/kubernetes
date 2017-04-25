package main

import (
	"flag"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/golang/glog"
	// For newer version of k8s use this package
	//	"k8s.io/apimachinery/pkg/util/uuid"

	aff "k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity"
	"k8s.io/kubernetes/cluster/addons/iso-client/discovery"
)

const (
	// kubelet eventDispatcher address
	eventDispatcherAddress = "localhost:5433"
	// iso-client own address
	eventHandlerLocalAddress = "localhost:5444"
	// name of isolator
	name = "iso"
)

// TODO: split it to smaller functions
func main() {
	flag.Parse()
	glog.Info("Starting ...")
	cpuTopo, err := discovery.DiscoverTopology()
	if err != nil {
		glog.Fatalf("Cannot retrive CPU topology: %q", err)
	}

	glog.Infof("Detected topology: %v", cpuTopo)

	var wg sync.WaitGroup
	// Starting eventHandlerServer
	server := aff.NewEventHandler(name, eventHandlerLocalAddress)
	err := server.RegisterEventHandler()
	if err != nil {
		glog.Fatalf("Cannot register EventHandler: %v", err)
		os.Exit(1)
	}
	wg.Add(1)
	go server.Serve(wg)

	// Sening address of local eventHandlerServer
	client, err := aff.NewEventDispatcherClient(name, eventDispatcherAddress, eventHandlerLocalAddress)
	if err != nil {
		glog.Fatalf("Cannot create eventDispatcherClient: %v", err)
		os.Exit(1)
	}

	reply, err := client.Register()
	if err != nil {
		glog.Fatalf("Failed to register handler: %v . Reply: %v", err, reply)
		os.Exit(1)
	}
	glog.Infof("Registering eventDispatcherClient. Reply: %v", reply)

	// Handling SigTerm
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go aff.HandleSIGTERM(c, client)

	wg.Wait()
}
