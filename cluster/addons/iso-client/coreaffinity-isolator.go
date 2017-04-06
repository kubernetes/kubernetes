package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"

	"github.com/golang/glog"
	coreaffiso "k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity"
	"k8s.io/kubernetes/cluster/addons/iso-client/isolator"
	opaq "k8s.io/kubernetes/cluster/addons/iso-client/opaque"
)

const (
	// kubelet eventDispatcher address
	eventDispatcherAddress = "localhost:5433"
	// iso-client own address
	isolatorLocalAddress = "localhost:5444"
	// name of isolator
	name = "coreaffinity-isolator"
)

func handleSIGTERM(sigterm chan os.Signal, client *isolator.EventDispatcherClient, opaque *opaq.OpaqueIntegerResourceAdvertiser, server *grpc.Server) {
	<-sigterm
	glog.Info("Received SIGTERM")
	shutdownIsolator(client, opaque, server)
}

func shutdownIsolator(client *isolator.EventDispatcherClient, opaque *opaq.OpaqueIntegerResourceAdvertiser, server *grpc.Server) {
	glog.Infof("Unregistering custom-isolator: %s", name)
	if err := client.UnregisterIsolator(); err != nil {
		opaque.RemoveOpaqueResource()
		glog.Errorf("Failed to unregister handler: %v")
	}

	glog.Infof("Removing opque integer resources %q from node %s", opaque.Name, opaque.Node)
	if err := opaque.RemoveOpaqueResource(); err != nil {
		glog.Fatalf("Failed to remove opaque resources: %v", err)
	}

	glog.Infof("%s has been unregistered", name)
	// send stop signal to grpc Server and exit isolator
	server.Stop()
}

func advertiseOpaqueCpus(cpus int) (*opaq.OpaqueIntegerResourceAdvertiser, error) {
	opaque, err := opaq.NewOpaqueIntegerResourceAdvertiser(name, fmt.Sprintf("%d", cpus))
	if err != nil {
		return nil, err
	}
	if err = opaque.AdvertiseOpaqueResource(); err != nil {
		return nil, err
	}
	return opaque, nil
}

// TODO: split it to smaller functions
func main() {
	flag.Parse()
	glog.Info("Starting ...")

	coreaffinityIsolator, err := coreaffiso.New(name)
	if err != nil {
		glog.Fatalf("Cannot create coreaffinity isolator: %q", err)
	}
	opaque, err := advertiseOpaqueCpus(coreaffinityIsolator.CPUTopology.GetTotalCPUs())
	if err != nil {
		glog.Fatalf("Cannot advertise opaque resources: %v", err)
	}

	// bind to socket
	socket, err := net.Listen("tcp", isolatorLocalAddress)
	if err != nil {
		opaque.RemoveOpaqueResource()
		glog.Fatalf("Cannot create tcp socket: %v", err)
	}

	finish := make(chan int)
	// Starting grpc server to handle isolation requests
	server := isolator.InitializeIsolatorServer(coreaffinityIsolator, socket)

	// Starting Isolator server in separeate goroutine
	go func(finish chan int) {
		defer func() { finish <- 1 }()
		if err := server.Serve(socket); err != nil {
			glog.Infof("Stopping isolator server: %v", err)
		}
	}(finish)

	glog.Info("Custom isolator server has been started")
	client, err := isolator.RegisterIsolator(name, eventDispatcherAddress, isolatorLocalAddress)
	if err != nil {
		opaque.RemoveOpaqueResource()
		server.Stop()
		glog.Fatalf("Failed to register eventDispatcher client: %v", err)
	}
	glog.Info("Coreaffinity isolator has been registered.")

	// Handling SigTerm
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go handleSIGTERM(c, client, opaque, server)
	<-finish
}
