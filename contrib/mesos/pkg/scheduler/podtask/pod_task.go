/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package podtask

import (
	"fmt"
	"strings"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/pborman/uuid"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	annotation "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/metrics"
	mresource "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resource"
	"k8s.io/kubernetes/pkg/api"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
)

type StateType int

const (
	StatePending StateType = iota
	StateRunning
	StateFinished
	StateUnknown
)

type FlagType string

const (
	Launched = FlagType("launched")
	Bound    = FlagType("bound")
	Deleted  = FlagType("deleted")
)

// A struct that describes a pod task.
type T struct {
	ID          string
	Pod         api.Pod
	Spec        Spec
	Offer       offers.Perishable // thread-safe
	State       StateType
	Flags       map[FlagType]struct{}
	CreateTime  time.Time
	UpdatedTime time.Time // time of the most recent StatusUpdate we've seen from the mesos master

	podStatus  api.PodStatus
	podKey     string
	launchTime time.Time
	bindTime   time.Time
	mapper     HostPortMappingType
}

type Spec struct {
	SlaveID       string
	AssignedSlave string
	CPU           mresource.CPUShares
	Memory        mresource.MegaBytes
	PortMap       []HostPortMapping
	Ports         []uint64
	Data          []byte
}

// mostly-clone this pod task. the clone will actually share the some fields:
//   - executor    // OK because it's read only
//   - Offer       // OK because it's guarantees safe concurrent access
func (t *T) Clone() *T {
	if t == nil {
		return nil
	}

	// shallow-copy
	clone := *t

	// deep copy
	(&t.Spec).copyTo(&clone.Spec)
	clone.Flags = map[FlagType]struct{}{}
	for k := range t.Flags {
		clone.Flags[k] = struct{}{}
	}
	return &clone
}

func (old *Spec) copyTo(new *Spec) {
	if len(old.PortMap) > 0 {
		new.PortMap = append(([]HostPortMapping)(nil), old.PortMap...)
	}
	if len(old.Ports) > 0 {
		new.Ports = append(([]uint64)(nil), old.Ports...)
	}
	if len(old.Data) > 0 {
		new.Data = append(([]byte)(nil), old.Data...)
	}
}

func (t *T) HasAcceptedOffer() bool {
	return t.Spec.SlaveID != ""
}

func (t *T) GetOfferId() string {
	if t.Offer == nil {
		return ""
	}
	return t.Offer.Details().Id.GetValue()
}

func generateTaskName(pod *api.Pod) string {
	ns := pod.Namespace
	if ns == "" {
		ns = api.NamespaceDefault
	}
	return fmt.Sprintf("%s.%s.pods", pod.Name, ns)
}

func setCommandArgument(ei *mesos.ExecutorInfo, flag, value string, create bool) {
	argv := []string{}
	overwrite := false
	if ei.Command != nil && ei.Command.Arguments != nil {
		argv = ei.Command.Arguments
		for i, arg := range argv {
			if strings.HasPrefix(arg, flag+"=") {
				overwrite = true
				argv[i] = flag + "=" + value
				break
			}
		}
	}
	if !overwrite && create {
		argv = append(argv, flag+"="+value)
		if ei.Command == nil {
			ei.Command = &mesos.CommandInfo{}
		}
		ei.Command.Arguments = argv
	}
}

func (t *T) BuildTaskInfo(prototype *mesos.ExecutorInfo) *mesos.TaskInfo {
	info := &mesos.TaskInfo{
		Name:     proto.String(generateTaskName(&t.Pod)),
		TaskId:   mutil.NewTaskID(t.ID),
		SlaveId:  mutil.NewSlaveID(t.Spec.SlaveID),
		Executor: proto.Clone(prototype).(*mesos.ExecutorInfo),
		Data:     t.Spec.Data,
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", float64(t.Spec.CPU)),
			mutil.NewScalarResource("mem", float64(t.Spec.Memory)),
		},
	}

	if portsResource := rangeResource("ports", t.Spec.Ports); portsResource != nil {
		info.Resources = append(info.Resources, portsResource)
	}

	// hostname needs of the executor needs to match that of the offer, otherwise
	// the kubelet node status checker/updater is very unhappy
	setCommandArgument(info.Executor, "--hostname-override", t.Spec.AssignedSlave, true)

	return info
}

// Clear offer-related details from the task, should be called if/when an offer
// has already been assigned to a task but for some reason is no longer valid.
func (t *T) Reset() {
	log.V(3).Infof("Clearing offer(s) from pod %v", t.Pod.Name)
	t.Offer = nil
	t.Spec = Spec{}
}

func (t *T) Set(f FlagType) {
	t.Flags[f] = struct{}{}
	if Launched == f {
		t.launchTime = time.Now()
		queueWaitTime := t.launchTime.Sub(t.CreateTime)
		metrics.QueueWaitTime.Observe(metrics.InMicroseconds(queueWaitTime))
	}
}

func (t *T) Has(f FlagType) (exists bool) {
	_, exists = t.Flags[f]
	return
}

func New(ctx api.Context, id string, pod *api.Pod) (*T, error) {
	key, err := MakePodKey(ctx, pod.Name)
	if err != nil {
		return nil, err
	}
	if id == "" {
		id = "pod." + uuid.NewUUID().String()
	}
	task := &T{
		ID:     id,
		Pod:    *pod,
		State:  StatePending,
		podKey: key,
		mapper: MappingTypeForPod(pod),
		Flags:  make(map[FlagType]struct{}),
	}
	task.CreateTime = time.Now()
	return task, nil
}

func (t *T) SaveRecoveryInfo(dict map[string]string) {
	dict[annotation.TaskIdKey] = t.ID
	dict[annotation.SlaveIdKey] = t.Spec.SlaveID
	dict[annotation.OfferIdKey] = t.Offer.Details().Id.GetValue()
}

// reconstruct a task from metadata stashed in a pod entry. there are limited pod states that
// support reconstruction. if we expect to be able to reconstruct state but encounter errors
// in the process then those errors are returned. if the pod is in a seemingly valid state but
// otherwise does not support task reconstruction return false. if we're able to reconstruct
// state then return a reconstructed task and true.
//
// at this time task reconstruction is only supported for pods that have been annotated with
// binding metadata, which implies that they've previously been associated with a task and
// that mesos knows about it.
//
// assumes that the pod data comes from the k8s registry and reflects the desired state.
//
func RecoverFrom(pod api.Pod) (*T, bool, error) {
	// we only expect annotations if pod has been bound, which implies that it has already
	// been scheduled and launched
	if pod.Spec.NodeName == "" && len(pod.Annotations) == 0 {
		log.V(1).Infof("skipping recovery for unbound pod %v/%v", pod.Namespace, pod.Name)
		return nil, false, nil
	}

	// only process pods that are not in a terminal state
	switch pod.Status.Phase {
	case api.PodPending, api.PodRunning, api.PodUnknown: // continue
	default:
		log.V(1).Infof("skipping recovery for terminal pod %v/%v", pod.Namespace, pod.Name)
		return nil, false, nil
	}

	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)
	key, err := MakePodKey(ctx, pod.Name)
	if err != nil {
		return nil, false, err
	}

	//TODO(jdef) recover ports (and other resource requirements?) from the pod spec as well

	now := time.Now()
	t := &T{
		Pod:        pod,
		CreateTime: now,
		podKey:     key,
		State:      StatePending, // possibly running? mesos will tell us during reconciliation
		Flags:      make(map[FlagType]struct{}),
		mapper:     MappingTypeForPod(&pod),
		launchTime: now,
		bindTime:   now,
	}
	var (
		offerId string
	)
	for _, k := range []string{
		annotation.BindingHostKey,
		annotation.TaskIdKey,
		annotation.SlaveIdKey,
		annotation.OfferIdKey,
	} {
		v, found := pod.Annotations[k]
		if !found {
			return nil, false, fmt.Errorf("incomplete metadata: missing value for pod annotation: %v", k)
		}
		switch k {
		case annotation.BindingHostKey:
			t.Spec.AssignedSlave = v
		case annotation.SlaveIdKey:
			t.Spec.SlaveID = v
		case annotation.OfferIdKey:
			offerId = v
		case annotation.TaskIdKey:
			t.ID = v
		}
	}
	t.Offer = offers.Expired(offerId, t.Spec.AssignedSlave, 0)
	t.Flags[Launched] = struct{}{}
	t.Flags[Bound] = struct{}{}
	return t, true, nil
}
