/*
Copyright 2016 The Kubernetes Authors.

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

package cm

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"

	"github.com/golang/glog"
	proto "github.com/golang/protobuf/proto"
	"github.com/pborman/uuid"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

// EventDispatcher manages a set of registered lifecycle event handlers and
// dispatches lifecycle events to them.
type EventDispatcher interface {
	// PreStartPod is invoked after the pod sandbox is created but before any
	// of a pod's containers are started.
	PreStartPod(pod *v1.Pod, cgroupPath string) (*lifecycle.EventReply, error)

	// PostStopPod is invoked after all of a pod's containers have permanently
	// stopped running, but before the pod sandbox is destroyed.
	PostStopPod(pod *v1.Pod, cgroupPath string) (*lifecycle.EventReply, error)

	// Start starts the dispatcher. After the dispatcher is started , handlers
	// can register themselves to receive lifecycle events.
	Start(socketAddress string)

	// Retrieving information about CgroupResources from replies
	ResourceConfigFromReplies(reply *lifecycle.EventReply, resources *ResourceConfig) *ResourceConfig
}

// Represents a registered event handler
type registeredHandler struct {
	// name by which the handler registered itself, unique
	name string
	// location of the event handler service.
	socketAddress string
	// token to identify this registration
	token string
}

type eventDispatcher struct {
	sync.Mutex
	started  bool
	handlers map[string]*registeredHandler
}

var dispatcher *eventDispatcher
var once sync.Once

func newEventDispatcher() *eventDispatcher {
	once.Do(func() {
		dispatcher = &eventDispatcher{
			handlers: map[string]*registeredHandler{},
		}
		dispatcher.Start(":5433") // "life" on a North American keypad
	})
	return dispatcher
}

func (ed *eventDispatcher) dispatchEvent(pod *v1.Pod, cgroupPath string, kind lifecycle.Event_Kind) (*lifecycle.EventReply, error) {
	jsonPod, err := json.Marshal(pod)
	if err != nil {
		return nil, err
	}
	// construct an event
	ev := &lifecycle.Event{
		Kind: kind,
		CgroupInfo: &lifecycle.CgroupInfo{
			Kind: lifecycle.CgroupInfo_POD,
			Path: cgroupPath,
		},
		Pod: jsonPod,
	}

	mergedReplies := &lifecycle.EventReply{}
	var errlist []error
	// TODO(CD): Re-evaluate nondeterministic delegation order arising
	//           from Go map iteration.
	for name, handler := range ed.handlers {
		// TODO(CD): Improve this by building a cancelable context
		ctx := context.Background()

		// Create a gRPC client connection
		// TODO(CD): Use SSL to connect to event handlers
		cxn, err := grpc.Dial(handler.socketAddress, grpc.WithInsecure())
		if err != nil {
			glog.Fatalf("failed to connect to event handler [%s] at [%s]: %v", handler.name, handler.socketAddress, err)
			errlist = append(errlist, err)
		}
		defer cxn.Close()
		client := lifecycle.NewEventHandlerClient(cxn)

		glog.Infof("Dispatching to event handler: %s", name)
		reply, err := client.Notify(ctx, ev)
		if err != nil {
			errlist = append(errlist, err)
		}
		proto.Merge(mergedReplies, reply)

	}
	return mergedReplies, utilerrors.NewAggregate(errlist)
}

func (ed *eventDispatcher) PreStartPod(pod *v1.Pod, cgroupPath string) (*lifecycle.EventReply, error) {
	return ed.dispatchEvent(pod, cgroupPath, lifecycle.Event_POD_PRE_START)
}

func (ed *eventDispatcher) PostStopPod(pod *v1.Pod, cgroupPath string) (*lifecycle.EventReply, error) {
	return ed.dispatchEvent(pod, cgroupPath, lifecycle.Event_POD_POST_STOP)
}

func (ed *eventDispatcher) Start(socketAddress string) {
	ed.Lock()
	defer ed.Unlock()

	if ed.started {
		glog.Info("event dispatcher is already running")
		return
	}

	glog.Infof("starting event dispatcher server at [%s]", socketAddress)

	// Set up server bind address.
	lis, err := net.Listen("tcp", socketAddress)
	if err != nil {
		glog.Fatalf("failed to bind to socket address: %v", err)
	}

	// Create a grpc.Server.
	s := grpc.NewServer()

	// Register self as KubeletEventDispatcherServer.
	lifecycle.RegisterEventDispatcherServer(s, ed)

	// Start listen in a separate goroutine.
	go func() {
		if err := s.Serve(lis); err != nil {
			glog.Fatalf("failed to start event dispatcher server: %v", err)
		}
	}()

	ed.started = true
}

func (ed *eventDispatcher) Register(ctx context.Context, request *lifecycle.RegisterRequest) (*lifecycle.RegisterReply, error) {
	// Create a registeredHandler instance
	h := &registeredHandler{
		name:          request.Name,
		socketAddress: request.SocketAddress,
		token:         uuid.NewUUID().String(),
	}

	glog.Infof("attempting to register event handler [%s]", h.name)

	// Check registered name for uniqueness
	reg := ed.handler(h.name)
	if reg != nil {
		if reg.token != request.Token {
			msg := fmt.Sprintf("registration failed: an event handler named [%s] is already registered and the supplied registration token does not match.", reg.name)
			glog.Warning(msg)
			return &lifecycle.RegisterReply{Error: msg}, nil
		}
		glog.Infof("re-registering event handler [%s]", h.name)
	}

	// Save registeredHandler
	ed.handlers[h.name] = h

	return &lifecycle.RegisterReply{Token: h.token}, nil
}

func (ed *eventDispatcher) Unregister(ctx context.Context, request *lifecycle.UnregisterRequest) (*lifecycle.UnregisterReply, error) {
	reg := ed.handler(request.Name)
	if reg == nil {
		msg := fmt.Sprintf("unregistration failed: no handler named [%s] is currently registered.", request.Name)
		glog.Warning(msg)
		return &lifecycle.UnregisterReply{Error: msg}, nil
	}
	if reg.token != request.Token {
		msg := fmt.Sprintf("unregistration failed: token mismatch for handler [%s].", request.Name)
		glog.Warning(msg)
		return &lifecycle.UnregisterReply{Error: msg}, nil
	}
	delete(ed.handlers, request.Name)
	return &lifecycle.UnregisterReply{}, nil
}

func (ed *eventDispatcher) ResourceConfigFromReplies(reply *lifecycle.EventReply, resources *ResourceConfig) *ResourceConfig {
	updatedResources := resources
	for _, cgroupResource := range reply.CgroupResource {
		if cgroupResource.CgroupSubsystem == lifecycle.CgroupResource_CPUSET_CPUS {
			updatedResources.CpusetCpus = &cgroupResource.Value
		}
	}
	return updatedResources
}

func (ed *eventDispatcher) handler(name string) *registeredHandler {
	for _, h := range ed.handlers {
		if h.name == name {
			return h
		}
	}
	return nil
}
