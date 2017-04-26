/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
	// PreStartPod is invoked before any of the pod's containers start.
	PreStartPod(podName string,
		containerName string,
		pod *v1.Pod,
		resource *lifecycle.CgroupInfo) ([]*lifecycle.IsolationControl, error)

	// PostStopPod is invoked after all of the pod's containers are terminated.
	PostStopPod(podName string, containerName string, cgroupInfo *lifecycle.CgroupInfo) error

	// PreStartContainer is invoked before there are any processes running inside
	// the container.
	PreStartContainer(podName string,
		containerName string) ([]*lifecycle.IsolationControl, error)

	// PostStopContainer is invoked after all of the processes in the container
	// are terminated.
	PostStopContainer(podName string,
		containerName string) error

	// ShutDown is invoked after the isolator is unregistered.
	ShutDown()

	// Perform any necessary pre-isolation initializations.
	Init() error

	// Returns the name of this isolator.
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

func (n NotifyHandler) preStartPod(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	pod, err := getPod(event.Pod)
	if err != nil {
		return &lifecycle.EventReply{
			Error:             err.Error(),
			IsolationControls: []*lifecycle.IsolationControl{},
		}, err
	}
	resources, err := n.isolator.PreStartPod(event.PodName, event.ContainerName, pod, event.CgroupInfo)
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

func (n NotifyHandler) postStopPod(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	if err := n.isolator.PostStopPod(event.PodName, event.ContainerName, event.CgroupInfo); err != nil {
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

func (n NotifyHandler) preStartContainer(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	resources, err := n.isolator.PreStartContainer(event.PodName, event.ContainerName)
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

func (n NotifyHandler) postStopContainer(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	if err := n.isolator.PostStopContainer(event.PodName, event.ContainerName); err != nil {
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
	case lifecycle.Event_CONTAINER_PRE_START:
		return n.preStartContainer(event)
	case lifecycle.Event_CONTAINER_POST_STOP:
		return n.postStopContainer(event)
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
