package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"google.golang.org/grpc"

	"github.com/golang/glog"
	coreaffiso "k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity"
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
	name = "coreaffinity-isolator"
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

func Serve(wg sync.WaitGroup, socket net.Listener, server *grpc.Server) {
	defer wg.Done()
	glog.Info("Starting serving")
	if err := server.Serve(socket); err != nil {
		glog.Fatalf("Coreaffinity isolator has stopped serving: %v", err)
	}
	glog.Info("Stopping coreaffinity isolator")
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

func registerEventDispatcherClient(name string, eventDispatcherAddress string, isolatorLocalAddress string) (*isolator.EventDispatcherClient, error) {
	client, err := isolator.NewEventDispatcherClient(name, eventDispatcherAddress, isolatorLocalAddress)
	if err != nil {
		return nil, fmt.Errorf("Cannot create eventDispatcherClient: %v", err)
	}

	// Registering to eventDispatcher server in kubelet
	reply, err := client.Register()
	if err != nil {
		return nil, fmt.Errorf("Failed to register isolator: %v . Reply: %v", err, reply)
	}

	return client, nil
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

	var wg sync.WaitGroup

	// create isolator Wrapper
	server := isolator.Register(coreaffinityIsolator)
	// bind to socket
	socket, err := net.Listen("tcp", isolatorLocalAddress)
	if err != nil {
		opaque.RemoveOpaqueResource()
		glog.Fatalf("Cannot create tcp socket: %v", err)
	}

	wg.Add(1)
	// Starting grpc server to handle isolation requests
	go Serve(wg, socket, server)

	client, err := registerEventDispatcherClient(name, eventDispatcherAddress, isolatorLocalAddress)
	if err != nil {
		opaque.RemoveOpaqueResource()
		glog.Fatalf("Failed to register eventDispatcher client: %v", err)
	}
	glog.Info("Coreaffinity isolator has been registered.")

	// Handling SigTerm
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go handleSIGTERM(c, client, opaque)

	wg.Wait()
}
