package main

import (
	"flag"
	"net"
	"os"
	"os/signal"
	"syscall"

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

	client, err := isolator.NewEventDispatcherClient(name, eventDispatcherAddress)
	if err != nil {
		// cleanup in case registration fail
		glog.Fatalf("Failed to create eventDispatcher client: %v", err)
	}
	glog.Infof("EventDispatcherClient has been created for %v", client.Name)
	// creating proper isolator
	coreaffinityIsolator, err := coreaffinity.New(name)
	if err != nil {
		glog.Fatalf("Cannot create coreaffinity isolator: %q", err)
	}

	// bind to socket
	socket, err := net.Listen("tcp", isolatorLocalAddress)
	if err != nil {
		glog.Fatalf("Cannot create tcp socket: %v", err)
	}
	// Initializing grpc server to handle isolation requests
	server := isolator.InitializeIsolatorServer(coreaffinityIsolator)

	finish := make(chan struct{})
	// Starting Isolator server in separeate goroutine
	go func() {
		defer close(finish)
		defer coreaffinityIsolator.ShutDown()
		if err := server.Serve(socket); err != nil {
			glog.Infof("Stopping isolator server: %v", err)
		}
	}()

	glog.Info("Custom isolator server has been started")
	// register isolator server to kubelet
	if err := client.RegisterIsolator(isolatorLocalAddress); err != nil {
		server.Stop()
		glog.Fatalf("Failed to register eventDispatcher client: %v", err)
	}

	// Handle cleanup in case of SIGTERM or error due to Registering the isolator
	sigterm := make(chan os.Signal, 2)
	signal.Notify(sigterm, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigterm
		if err := client.UnregisterIsolator(); err != nil {
			glog.Errorf("Failed to unregister handler: %v")
		}
		glog.Infof("%s has been unregistered", name)
		server.Stop()
	}()

	glog.Info("Coreaffinity isolator has been registered.")

	// Handling SIGTerm
	// wait till grpc Server is stopped
	<-finish
}
