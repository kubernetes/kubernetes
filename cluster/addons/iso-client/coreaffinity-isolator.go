package main

import (
	"flag"
	"os"
	"os/signal"
	"sync"
	"syscall"

	// For newer version of k8s use this package
	//	"k8s.io/apimachinery/pkg/util/uuid"

	"fmt"
	"github.com/golang/glog"
	aff "k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity"
	"k8s.io/kubernetes/cluster/addons/iso-client/discovery"
	opaq "k8s.io/kubernetes/cluster/addons/iso-client/opaque"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

const (
	// kubelet eventDispatcher address
	eventDispatcherAddress = "localhost:5433"
	// iso-client own address
	isolatorLocalAddress = "localhost:5444"
	// name of isolator
	name = "iso"
)

func handleSIGTERM(sigterm chan os.Signal, client *aff.EventDispatcherClient, opaque *opaq.OpaqueIntegerResourceAdvertiser) {
	<-sigterm
	shutdownIsolator(client, opaque)
}

func shutdownIsolator(client *aff.EventDispatcherClient, opaque *opaq.OpaqueIntegerResourceAdvertiser) {
	unregisterRequest := &lifecycle.UnregisterRequest{
		Name:  client.Name,
		Token: client.Token,
	}

	if _, err := client.Unregister(client.Ctx, unregisterRequest); err != nil {
		opaque.RemoveOpaqueResource()
		glog.Fatalf("Failed to unregister handler: %v")
	}
	glog.Infof("Unregistering custom-isolator: %s", name)

	if err := opaque.RemoveOpaqueResource(); err != nil {
		glog.Fatalf("Failed to remove opaque resources: %v", err)
	}

	os.Exit(0)

}

// TODO: split it to smaller functions
func main() {
	flag.Parse()
	glog.Info("Starting ...")
	topology, err := discovery.DiscoverTopology()
	if err != nil {
		glog.Fatalf("Cannot retrive CPU topology: %q", err)
	}
	glog.Infof("Detected topology: %v", topology)

	opaque, err := opaq.NewOpaqueIntegerResourceAdvertiser(name, fmt.Sprintf("%d", topology.GetTotalCPUs()))
	if err != nil {
		shutdownIsolator(nil, opaque)
		glog.Fatalf("Cannot create opaque resource advertiser: %v", err)
	}
	if err = opaque.AdvertiseOpaqueResource(); err != nil {
		shutdownIsolator(nil, opaque)
		glog.Fatalf("Failed to advertise opaque resources: %v", err)
	}

	var wg sync.WaitGroup
	// Starting isolatorServer
	server := aff.NewIsolator(name, isolatorLocalAddress, topology)
	err = server.RegisterIsolator()
	if err != nil {
		shutdownIsolator(nil, opaque)
		glog.Fatalf("Cannot register isolator: %v", err)
	}
	wg.Add(1)
	go server.Serve(wg)

	// Sending address of local isolatorServer
	client, err := aff.NewEventDispatcherClient(name, eventDispatcherAddress, isolatorLocalAddress)
	if err != nil {
		shutdownIsolator(client, opaque)
		glog.Fatalf("Cannot create eventDispatcherClient: %v", err)
		os.Exit(1)
	}

	reply, err := client.Register()
	if err != nil {
		shutdownIsolator(client, opaque)
		glog.Fatalf("Failed to register isolator: %v . Reply: %v", err, reply)
		os.Exit(1)
	}
	glog.Infof("Registering isolator. Reply: %v", reply)

	// Handling SigTerm
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go handleSIGTERM(c, client, opaque)

	wg.Wait()
}
