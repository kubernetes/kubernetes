package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

const (
	// kubelet eventDispatcher address
	eventDispatcherAddress = "localhost:5433"
	// iso-client own address
	eventHandlerAddress = "localhost:5444"
	// name of isolator
	name = "iso"
)

type EventHandler interface {
	Start(socketAddress string)
}

type eventHandler struct {
}

// starting eventHandler grpc server
func (e *eventHandler) Start(socketAddress string) {
	glog.Infof("Starting eventHandler")
	lis, err := net.Listen("tcp", socketAddress)
	if err != nil {
		glog.Fatalf("failed to bind to socket address: %v", err)
	}
	s := grpc.NewServer()

	lifecycle.RegisterEventHandlerServer(s, e)

	if err := s.Serve(lis); err != nil {
		glog.Fatalf("failed to start event handler server: %v", err)
	}
}

// TODO: implement PostStop
func (e *eventHandler) Notify(context context.Context, event *lifecycle.Event) (reply *lifecycle.EventReply, err error) {
	switch event.Kind {
	case lifecycle.Event_POD_PRE_START:
		glog.Infof("Received PreStart event with such payload: %v\n", event.CgroupInfo)
		// Pinning created pod to static 0-1 core
		path := fmt.Sprintf("%s%s%s", "/sys/fs/cgroup/cpuset", event.CgroupInfo.Path, "/cpuset.cpus")
		glog.Infof("Our path: %v", path)
		err := ioutil.WriteFile(path, []byte("0-1"), 0644)
		if err != nil {
			return nil, fmt.Errorf("Ooops: %v", err)
		}

		return &lifecycle.EventReply{
			Error:      "",
			CgroupInfo: event.CgroupInfo,
		}, nil

	default:
		return nil, fmt.Errorf("Wrong event type")
	}

}

// TODO: split it to smaller functions
func main() {
	flag.Parse()
	glog.Info("Staring ...")
	cxn, err := grpc.Dial(eventDispatcherAddress, grpc.WithInsecure())
	if err != nil {
		glog.Fatalf("failed to connect to eventDispatcher: %v", err)
	}
	client := lifecycle.NewEventDispatcherClient(cxn)
	glog.Infof("Registering handler: %s\n", name)

	registerToken := string(uuid.NewUUID())
	registerRequest := &lifecycle.RegisterRequest{
		SocketAddress: eventHandlerAddress,
		Name:          name,
		Token:         registerToken,
	}

	ctx := context.Background()
	reply, err := client.Register(ctx, registerRequest)
	if err != nil {
		glog.Fatalf("Failed to register handler: %v")
	}

	glog.Infof("Registered iso: %v\n", reply)
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		unregisterRequest := &lifecycle.UnregisterRequest{
			Name:  name,
			Token: reply.Token,
		}
		rep, err := client.Unregister(ctx, unregisterRequest)
		if err != nil {
			glog.Fatalf("Failed to unregister handler: %v")
		}
		glog.Infof("Unregistered iso: %v\n", rep)

		os.Exit(0)
	}()
	// TODO: Move it to separete goroutine to avoid races and start it before registering
	server := &eventHandler{}
	server.Start(eventHandlerAddress)

}
