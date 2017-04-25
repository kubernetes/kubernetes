package coreaffinity

import (
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/cluster/addons/iso-client/cputopology"
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
	Name             string
	Address          string
	GrpcServer       *grpc.Server
	Socket           net.Listener
	CPUTopology      *cputopology.CPUTopology
	cpuAssignmentMap map[string][]int
}

// Constructor for Isolator
func NewIsolator(isolatorName string, isolatorAddress string, topology *cputopology.CPUTopology) (e *isolator) {
	return &isolator{
		Name:             isolatorName,
		Address:          isolatorAddress,
		CPUTopology:      topology,
		cpuAssignmentMap: make(map[string][]int),
	}
}

// Registering isolator server
func (i *isolator) RegisterIsolator() (err error) {
	i.Socket, err = net.Listen("tcp", i.Address)
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

func (i *isolator) gatherContainerRequest(container v1.Container) int64 {
	resource, ok := container.Resources.Requests[v1.OpaqueIntResourceName(i.Name)]
	if !ok {
		return 0
	}
	return resource.Value()
}

func (i *isolator) countCoresFromOIR(pod *v1.Pod) int64 {
	var coresAccu int64
	for _, container := range pod.Spec.Containers {
		coresAccu = coresAccu + i.gatherContainerRequest(container)
	}
	return coresAccu
}

func (i *isolator) reserveCPUs(cores int64) ([]int, error) {
	cpus := i.CPUTopology.GetAvailableCPUs()
	if len(cpus) < int(cores) {
		return nil, fmt.Errorf("cannot reserved requested number of cores")
	}
	var reservedCores []int

	for idx := 0; idx < int(cores); idx++ {
		if err := i.CPUTopology.Reserve(cpus[idx]); err != nil {
			return reservedCores, err
		}
		reservedCores = append(reservedCores, cpus[idx])
	}
	return reservedCores, nil

}

func (i *isolator) reclaimCPUs(cores []int) {
	for _, core := range cores {
		i.CPUTopology.Reclaim(core)
	}
}

func asCPUList(cores []int) string {
	var coresStr []string
	for _, core := range cores {
		coresStr = append(coresStr, strconv.Itoa(core))
	}
	return strings.Join(coresStr, ",")
}

func (i *isolator) preStart(event *lifecycle.Event) (reply *lifecycle.EventReply, err error) {
	glog.Infof("available cores before %v", i.CPUTopology)
	glog.Infof("Received PreStart event: %v\n", event.CgroupInfo)
	pod, err := getPod(event.Pod)
	if err != nil {
		return &lifecycle.EventReply{
			Error:          err.Error(),
			CgroupInfo:     event.CgroupInfo,
			CgroupResource: []*lifecycle.CgroupResource{},
		}, err
	}
	oirCores := i.countCoresFromOIR(pod)
	glog.Infof("Pod %s requested %d cores", pod.Name, oirCores)

	if oirCores == 0 {
		glog.Infof("Pod %q isn't managed by this isolator", pod.Name)
		return &lifecycle.EventReply{
			Error:          "",
			CgroupInfo:     event.CgroupInfo,
			CgroupResource: []*lifecycle.CgroupResource{},
		}, nil
	}

	reservedCores, err := i.reserveCPUs(oirCores)
	if err != nil {
		i.reclaimCPUs(reservedCores)
		return &lifecycle.EventReply{
			Error:          err.Error(),
			CgroupInfo:     event.CgroupInfo,
			CgroupResource: []*lifecycle.CgroupResource{},
		}, err
	}

	cgroupResource := []*lifecycle.CgroupResource{
		{
			Value:           asCPUList(reservedCores),
			CgroupSubsystem: lifecycle.CgroupResource_CPUSET_CPUS,
		},
	}
	i.cpuAssignmentMap[event.CgroupInfo.Path] = reservedCores
	glog.Infof("CPU MAP: %v", i.cpuAssignmentMap)
	glog.Infof("cores %v", asCPUList(reservedCores))
	glog.Infof("available cores after %v", i.CPUTopology)

	return &lifecycle.EventReply{
		Error:          "",
		CgroupInfo:     event.CgroupInfo,
		CgroupResource: cgroupResource,
	}, nil
}

func (i *isolator) postStop(event *lifecycle.Event) (reply *lifecycle.EventReply, err error) {
	glog.Infof("Event: %v", event)
	cpus := i.cpuAssignmentMap[event.CgroupInfo.Path]
	i.reclaimCPUs(cpus)
	delete(i.cpuAssignmentMap, event.CgroupInfo.Path)

	glog.Infof("CPUs: %v have been revoked.", cpus)
	glog.Infof("available cores after %v", i.CPUTopology)
	return &lifecycle.EventReply{
		Error:          "",
		CgroupInfo:     event.CgroupInfo,
		CgroupResource: []*lifecycle.CgroupResource{},
	}, nil

}

// TODO: implement PostStop
func (i *isolator) Notify(context context.Context, event *lifecycle.Event) (reply *lifecycle.EventReply, err error) {
	switch event.Kind {
	case lifecycle.Event_POD_PRE_START:
		return i.preStart(event)
	case lifecycle.Event_POD_POST_STOP:
		return i.postStop(event)
	default:
		return nil, fmt.Errorf("Wrong event type")
	}

}
