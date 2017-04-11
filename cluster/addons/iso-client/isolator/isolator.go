package isolator

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/golang/glog"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type Isolator interface {
	// preStart Hook to be implemented by custom isolators
	PreStartPod(pod *v1.Pod, resource *lifecycle.CgroupInfo) ([]*lifecycle.IsolationControl, error)
	// postStop Hook to be implemented by custom isolators
	PostStopPod(cgroupInfo *lifecycle.CgroupInfo) error
	// cleanUp after isolator is turned off
	ShutDown()
	// Initialize resources when connection to eventDispatcher is possible, before registering the isolator in kubelet
	Init() error
	// get Name of isolator
	Name() string
}

// wrapper for custom isolators which implements Isolator protobuf service with Notify method
type NotifyHandler struct {
	isolator Isolator
}

// extract Pod object from Event
// @pre: bytePod is a UTF-8 string containing a JSON-encoded pod object.
func getPod(bytePod []byte) (*v1.Pod, error) {
	pod := &v1.Pod{}
	if err := json.Unmarshal(bytePod, pod); err != nil {
		return nil, fmt.Errorf("Cannot unamrshall POD: %v", err)
	}
	return pod, nil
}

// wrapper for preStart method
func (n NotifyHandler) preStartPod(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	pod, err := getPod(event.Pod)
	if err != nil {
		return &lifecycle.EventReply{
			Error:             err.Error(),
			IsolationControls: []*lifecycle.IsolationControl{},
		}, err
	}
	resources, err := n.isolator.PreStartPod(pod, event.CgroupInfo)
	if err != nil {
		return &lifecycle.EventReply{
			Error:             err.Error(),
			IsolationControls: []*lifecycle.IsolationControl{},
		}, err
	}
	return &lifecycle.EventReply{
		Error:             "",
		IsolationControls: resources,
	}, nil
}

// wrapper for postStop method
func (n NotifyHandler) postStopPod(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	if err := n.isolator.PostStopPod(event.CgroupInfo); err != nil {
		return &lifecycle.EventReply{
			Error:             err.Error(),
			IsolationControls: []*lifecycle.IsolationControl{},
		}, err
	}
	return &lifecycle.EventReply{
		Error:             "",
		IsolationControls: []*lifecycle.IsolationControl{},
	}, nil
}

func (n NotifyHandler) Notify(context context.Context, event *lifecycle.Event) (*lifecycle.EventReply, error) {
	switch event.Kind {
	case lifecycle.Event_POD_PRE_START:
		return n.preStartPod(event)
	case lifecycle.Event_POD_POST_STOP:
		return n.postStopPod(event)
	default:
		glog.Infof("Unknown event type: %v", event.Kind)
		return nil, nil
	}
}

// Start grpc server and handle shutdown
// Blocking
// In case of error ShutDown() function of "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle".Isolator  is invoked
func StartIsolatorServer(i Isolator, isolatorLocalAddress string, eventDispatcherAddress string) error {
	stopped, stopServer, err := startServer(i, isolatorLocalAddress)
	if err != nil {
		return err
	}
	if err = i.Init(); err != nil {
		return err
	}
	client, err := registerClient(i.Name(), eventDispatcherAddress, isolatorLocalAddress)
	if err != nil {
		i.ShutDown()
		stopServer()
		return err
	}
	// Handle cleanup in case of SIGTERM or error due to Registering the isolator
	sigterm := make(chan os.Signal, 2)
	signal.Notify(sigterm, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigterm
		if err := client.UnregisterIsolator(); err != nil {
			glog.Errorf("Failed to unregister handler: %v")
		}
		i.ShutDown()
		stopServer()
		glog.Infof("%s has been unregistered", i.Name())
	}()
	<-stopped
	return nil
}

func startServer(i Isolator, isolatorLocalAddress string) (chan struct{}, func(), error) {
	grpcServer := grpc.NewServer()
	// create wrapper
	nh := &NotifyHandler{isolator: i}
	// register grpc server implementing Notify() method
	lifecycle.RegisterIsolatorServer(grpcServer, nh)

	// bind to socket
	socket, err := net.Listen("tcp", isolatorLocalAddress)
	if err != nil {
		return nil, nil, fmt.Errorf("Cannot create tcp socket: %v", err)
	}

	stopped := make(chan struct{})
	// starting server
	go func() {
		defer close(stopped)
		if err := grpcServer.Serve(socket); err != nil {
			glog.Infof("Stopping isolator server: %v", err)
		}
	}()

	return stopped, grpcServer.Stop, nil
}

func registerClient(isolatorName, eventDispatcherAddress, isolatorLocalAddress string) (*EventDispatcherClient, error) {
	// create a grpc client to connect to eventDispatcher
	client, err := newEventDispatcherClient(isolatorName, eventDispatcherAddress)
	if err != nil {
		return nil, fmt.Errorf("Failed to create eventDispatcher client: %v", err)
	}

	// register isolator server to kubelet
	if err := client.RegisterIsolator(isolatorLocalAddress); err != nil {
		return nil, fmt.Errorf("Failed to register eventDispatcher client: %v", err)
	}
	return client, err
}
