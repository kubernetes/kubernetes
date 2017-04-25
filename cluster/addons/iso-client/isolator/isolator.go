package isolator

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type Isolator interface {
	PreStart(pod *v1.Pod, resource *lifecycle.CgroupInfo) ([]*lifecycle.CgroupResource, error)
	PostStop(cgroupInfo *lifecycle.CgroupInfo) error
}

type NotifyHandler struct {
	isolator Isolator
}

func NewNotifyHandler(i Isolator) *NotifyHandler {
	return &NotifyHandler{isolator: i}
}

// extract Pod object from Event
func getPod(bytePod []byte) (pod *v1.Pod, err error) {
	pod = &v1.Pod{}
	err = json.Unmarshal(bytePod, pod)
	if err != nil {
		glog.Fatalf("Cannot Unmarshal pod: %v", err)
		return
	}
	return
}

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

func Register(nh *NotifyHandler) *grpc.Server {
	grpcServer := grpc.NewServer()
	lifecycle.RegisterIsolatorServer(grpcServer, nh)
	glog.Info("Isolator Server has been registered")

	return grpcServer
}

func Serve(wg sync.WaitGroup, socket net.Listener, grpcServer *grpc.Server) {
	defer wg.Done()
	glog.Info("Starting serving")
	if err := grpcServer.Serve(socket); err != nil {
		glog.Fatalf("Isolator server stopped serving : %v", err)
	}
	glog.Info("Stopping isolatorServer")
}
