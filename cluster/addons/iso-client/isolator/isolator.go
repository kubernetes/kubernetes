package isolator

import (
	"encoding/json"
	"fmt"

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
	// cleanUP after isolator is turned off
	ShutDown()
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

func InitializeIsolatorServer(i Isolator) *grpc.Server {
	// create wrapper
	nh := &NotifyHandler{isolator: i}
	grpcServer := grpc.NewServer()
	// register grpc server implementing Notify() method
	lifecycle.RegisterIsolatorServer(grpcServer, nh)
	return grpcServer
}
