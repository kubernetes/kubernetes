package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/golang/glog"
	"google.golang.org/grpc"
	coreaffiso "k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity"
	"k8s.io/kubernetes/cluster/addons/iso-client/isolator"
	opaque "k8s.io/kubernetes/cluster/addons/iso-client/opaque"
)

const (
	// kubelet eventDispatcher address
	eventDispatcherAddress = "localhost:5433"
	// iso-client own address
	isolatorLocalAddress = "localhost:5444"
	// name of isolator
	name = "coreaffinity-isolator"
)

func handleSIGTERM(sigterm chan os.Signal, client *isolator.EventDispatcherClient, server *grpc.Server) {
	<-sigterm
	glog.Info("Received SIGTERM")
	shutdownIsolator(client, server)
}

func shutdownIsolator(client *isolator.EventDispatcherClient, server *grpc.Server) {
	if err := client.UnregisterIsolator(); err != nil {
		glog.Errorf("Failed to unregister handler: %v")
	}
	glog.Infof("%s has been unregistered", name)

	opaque.RemoveOpaqueResource(name)
	// send stop signal to grpc Server and exit isolator
	server.Stop()
}

// TODO: split it to smaller functions
func main() {
	flag.Parse()
	glog.Info("Starting ...")

	// creating proper isolator
	coreaffinityIsolator, err := coreaffiso.New(name)
	if err != nil {
		glog.Fatalf("Cannot create coreaffinity isolator: %q", err)
	}
	// bind to socket
	socket, err := net.Listen("tcp", isolatorLocalAddress)
	if err != nil {
		glog.Fatalf("Cannot create tcp socket: %v", err)
	}

	finish := make(chan struct{})
	// Initializing grpc server to handle isolation requests
	server := isolator.InitializeIsolatorServer(coreaffinityIsolator, socket)

	// Starting Isolator server in separeate goroutine
	go func() {
		defer close(finish)
		if err := server.Serve(socket); err != nil {
			glog.Infof("Stopping isolator server: %v", err)
		}
	}()

	// advertise opaque resources
	if err := opaque.AdvertiseOpaqueResource(name, fmt.Sprintf("%d", coreaffinityIsolator.CPUTopology.GetTotalCPUs())); err != nil {
		glog.Fatalf("Cannot advertise opaque resources: %v", err)
	}
	glog.Info("Custom isolator server has been started")

	// register isolator server to kubelet
	client, err := isolator.RegisterIsolator(name, eventDispatcherAddress, isolatorLocalAddress)
	if err != nil {
		// cleanup in case registration fail
		opaque.RemoveOpaqueResource(name)
		server.Stop()
		glog.Fatalf("Failed to register eventDispatcher client: %v", err)
	}
	glog.Info("Coreaffinity isolator has been registered.")

	// Handling SIGTerm
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go handleSIGTERM(c, client, server)

	// wait till grpc Server is stopped
	<-finish
}
