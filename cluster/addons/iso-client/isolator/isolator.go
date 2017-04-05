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
	PreStart(pod *v1.Pod, resource *lifecycle.CgroupInfo) ([]*lifecycle.CgroupResource, error)
	// postStop Hook to be implemented by custom isolators
	PostStop(cgroupInfo *lifecycle.CgroupInfo) error
}

// wrapper for custom isolators which implements Isolator protobuf service with Notify method
type NotifyHandler struct {
	isolator Isolator
}

// extract Pod object from Event
func getPod(bytePod []byte) (*v1.Pod, error) {
	pod := &v1.Pod{}
	if err := json.Unmarshal(bytePod, pod); err != nil {
		return nil, fmt.Errorf("Cannot unamrshall POD: %v", err)
	}
	return pod, nil
}

// wrapper for preStart method
func (n NotifyHandler) preStart(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	pod, err := getPod(event.Pod)
	if err != nil {
		return &lifecycle.EventReply{
			Error:          err.Error(),
			CgroupInfo:     event.CgroupInfo,
			CgroupResource: []*lifecycle.CgroupResource{},
		}, err
	}
	resources, err := n.isolator.PreStart(pod, event.CgroupInfo)
	if err != nil {
		return &lifecycle.EventReply{
			Error:          err.Error(),
			CgroupInfo:     event.CgroupInfo,
			CgroupResource: []*lifecycle.CgroupResource{},
		}, err
	}
	return &lifecycle.EventReply{
		Error:          "",
		CgroupInfo:     event.CgroupInfo,
		CgroupResource: resources,
	}, nil
}

// wrapper for postStop method
func (n NotifyHandler) postStop(event *lifecycle.Event) (*lifecycle.EventReply, error) {
	if err := n.isolator.PostStop(event.CgroupInfo); err != nil {
		return &lifecycle.EventReply{
			Error:          err.Error(),
			CgroupInfo:     event.CgroupInfo,
			CgroupResource: []*lifecycle.CgroupResource{},
		}, err
	}
	return &lifecycle.EventReply{
		Error:          "",
		CgroupInfo:     event.CgroupInfo,
		CgroupResource: []*lifecycle.CgroupResource{},
	}, nil
}

func (n NotifyHandler) Notify(context context.Context, event *lifecycle.Event) (*lifecycle.EventReply, error) {
	switch event.Kind {
	case lifecycle.Event_POD_PRE_START:
		return n.preStart(event)
	case lifecycle.Event_POD_POST_STOP:
		return n.postStop(event)
	default:
		return nil, fmt.Errorf("Wrong event type")
	}
}

func Register(i Isolator) *grpc.Server {
	nh := &NotifyHandler{isolator: i}
	grpcServer := grpc.NewServer()
	lifecycle.RegisterIsolatorServer(grpcServer, nh)
	glog.Info("Isolator Server has been registered")
	return grpcServer
}
