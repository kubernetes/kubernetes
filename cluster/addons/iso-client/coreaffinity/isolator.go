package coreaffinity

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"sync"

	"golang.org/x/net/context"

	"github.com/golang/glog"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

const (
	cgroupPrefix = "/sys/fs/cgroup/cpuset"
	cgroupSufix  = "/cpuset.cpus"
)

type Isolator interface {
	RegisterIsolator() error
	Serve(sync.WaitGroup)
}

type isolator struct {
	Name       string
	Address    string
	GrpcServer *grpc.Server
	Socket     net.Listener
}

// Constructor for Isolator
func NewIsolator(isolatorName string, isolatorAddress string) (e *isolator) {
	return &isolator{
		Name:    isolatorName,
		Address: isolatorAddress,
	}
}

// Registering isolator server
func (i *isolator) RegisterIsolator() (err error) {
	i.Socket, err = net.Listen("tcp", e.Address)
	if err != nil {
		return fmt.Errorf("Failed to bind to socket address: %v", err)
	}
	i.GrpcServer = grpc.NewServer()

	lifecycle.RegisterIsolatorServer(i.GrpcServer, i)
	glog.Info("Isolator Server has been registered")

	return nil

}

// Start serving Grpc
func (i *isolator) Serve(wg sync.WaitGroup) {
	defer wg.Done()
	glog.Info("Starting serving")
	if err := i.GrpcServer.Serve(i.Socket); err != nil {
		glog.Fatalf("Isolator server stopped serving : %v", err)
	}
	glog.Info("Stopping isolatorServer")
}

// isolation api
type isoSpec struct {
	CoreAffinity string `json:"core-affinity"`
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

// check wheter pod should be isolated
func isIsoPod(pod *v1.Pod) bool {
	if pod.Annotations["pod.alpha.kubernetes.io/isolation-api"] != "" {
		return true
	}
	return false
}

// extract isoSpec from annotations
func getIsoSpec(annotations map[string]string) (spec *isoSpec, err error) {
	spec = &isoSpec{}
	err = json.Unmarshal([]byte(annotations["pod.alpha.kubernetes.io/isolation-api"]), spec)
	if err != nil {
		glog.Fatalf("Cannot unmarshal isoSpec: %v", err)
		return nil, err
	}
	return
}

// validate isoSpec
func validateIsoSpec(spec *isoSpec) (err error) {
	if spec.CoreAffinity == "" {
		return fmt.Errorf("Required field core-affinity is missing.")
	}
	return nil
}

// TODO: implement PostStop
func (i *isolator) Notify(context context.Context, event *lifecycle.Event) (reply *lifecycle.EventReply, err error) {
	switch event.Kind {
	case lifecycle.Event_POD_PRE_START:
		glog.Infof("Received PreStart event: %v\n", event.CgroupInfo)
		pod, err := getPod(event.Pod)
		if err != nil {
			return &lifecycle.EventReply{
				Error:      err.Error(),
				CgroupInfo: event.CgroupInfo,
			}, err
		}

		if !isIsoPod(pod) {
			glog.Infof("Pod %s is not managed by this isolator", pod.Name)
			return &lifecycle.EventReply{
				Error:      "",
				CgroupInfo: event.CgroupInfo,
			}, nil
		}

		glog.Infof("Pod %s is managed by this isolator", pod.Name)
		spec, err := getIsoSpec(pod.Annotations)
		if err != nil {
			return &lifecycle.EventReply{
				Error:      err.Error(),
				CgroupInfo: event.CgroupInfo,
			}, nil
		}
		// TODO: Decide whether typo should error POD or not
		err = validateIsoSpec(spec)
		if err != nil {
			return &lifecycle.EventReply{
				Error:      fmt.Sprintf("Spec is not valid. Given json: %v", pod.Annotations["pod.alpha.kubernetes.io/isolation-api"]),
				CgroupInfo: event.CgroupInfo,
			}, nil
		}
		glog.Infof("Pod %s is valid. Value of core-affinity: %s", pod.Name, spec.CoreAffinity)

		return &lifecycle.EventReply{
			Error:      "",
			CgroupInfo: event.CgroupInfo,
			CgroupResource: &lifecycle.CgroupResource{
				Value:           spec.CoreAffinity,
				CgroupSubsystem: lifecycle.CgroupResource_CPUSET_CPUS,
			},
		}, nil
	default:
		return nil, fmt.Errorf("Wrong event type")
	}

}
