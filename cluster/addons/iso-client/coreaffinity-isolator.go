package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/golang/glog"
	aff "k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity"
	"k8s.io/kubernetes/cluster/addons/iso-client/isolator"
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

func handleSIGTERM(sigterm chan os.Signal, client *isolator.EventDispatcherClient, opaque *opaq.OpaqueIntegerResourceAdvertiser) {
	<-sigterm
	shutdownIsolator(client, opaque)
}

func shutdownIsolator(client *isolator.EventDispatcherClient, opaque *opaq.OpaqueIntegerResourceAdvertiser) {
	unregisterRequest := &lifecycle.UnregisterRequest{
		Name:  client.Name,
	}

	glog.Infof("Unregistering custom-isolator: %s", name)
	if _, err := client.Unregister(client.Ctx, unregisterRequest); err != nil {
		opaque.RemoveOpaqueResource()
		glog.Fatalf("Failed to unregister handler: %v")
	}

	glog.Infof("Removing opque integer resources %q from node %s", opaque.Name, opaque.Node)
	if err := opaque.RemoveOpaqueResource(); err != nil {
		glog.Fatalf("Failed to remove opaque resources: %v", err)
	}

	glog.Infof("%s has been unregistered", name)
	os.Exit(0)

}

// TODO: split it to smaller functions
func main() {
	flag.Parse()
	glog.Info("Starting ...")

	cpuIsolator, err := aff.NewIsolator(name)
	if err != nil {
		glog.Fatalf("Cannot create coreaffinity isolator: %q", err)
	}

	opaque, err := opaq.NewOpaqueIntegerResourceAdvertiser(name, fmt.Sprintf("%d", cpuIsolator.CPUTopology.GetTotalCPUs()))
	if err != nil {
		glog.Errorf("Cannot create opaque resource advertiser: %v", err)
		shutdownIsolator(nil, opaque)
	}
	if err = opaque.AdvertiseOpaqueResource(); err != nil {
		glog.Errorf("Failed to advertise opaque resources: %v", err)
		shutdownIsolator(nil, opaque)
	}

	var wg sync.WaitGroup
	// Starting isolatorServer
	server := isolator.NewNotifyHandler(cpuIsolator)
	grpcServer := isolator.Register(server)
	socket, err := net.Listen("tcp", isolatorLocalAddress)
	if err != nil {
		glog.Errorf("Cannot create tcp socket: %v", err)
		shutdownIsolator(nil, opaque)
	}

	wg.Add(1)
	go isolator.Serve(wg, socket, grpcServer)

	// Sending address of local isolatorServer
	client, err := isolator.NewEventDispatcherClient(name, eventDispatcherAddress, isolatorLocalAddress)
	if err != nil {
		glog.Errorf("Cannot create eventDispatcherClient: %v", err)
		shutdownIsolator(client, opaque)
	}

	reply, err := client.Register()
	if err != nil {
		glog.Errorf("Failed to register isolator: %v . Reply: %v", err, reply)
		shutdownIsolator(client, opaque)
	}
	glog.Infof("Registering isolator. Reply: %v", reply)

	// Handling SigTerm
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go handleSIGTERM(c, client, opaque)

	wg.Wait()
}
