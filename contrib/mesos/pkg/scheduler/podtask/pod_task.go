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
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/metrics"
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

var defaultRoles = []string{"*"}

// A struct that describes a pod task.
type T struct {
	ID  string
	Pod api.Pod

	State       StateType
	Flags       map[FlagType]struct{}
	CreateTime  time.Time
	UpdatedTime time.Time // time of the most recent StatusUpdate we've seen from the mesos master

	podStatus    api.PodStatus
	prototype    *mesos.ExecutorInfo // readonly
	allowedRoles []string            // roles under which pods are allowed to be launched
	podKey       string
	launchTime   time.Time
	bindTime     time.Time
	mapper       HostPortMapper
}

type Port struct {
	Port uint64
	Role string
}

type Spec struct {
	SlaveID       string
	AssignedSlave string
	Resources     []*mesos.Resource
	PortMap       []HostPortMapping
	Data          []byte
	Executor      *mesos.ExecutorInfo
	OfferID       *mesos.OfferID
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
	clone.Flags = map[FlagType]struct{}{}
	for k := range t.Flags {
		clone.Flags[k] = struct{}{}
	}
	return &clone
}

func generateTaskName(pod *api.Pod) string {
	ns := pod.Namespace
	if ns == "" {
		ns = api.NamespaceDefault
	}
	return fmt.Sprintf("%s.%s.pods", pod.Name, ns)
}

func (t *T) BuildTaskInfo(spec *Spec) (*mesos.TaskInfo, error) {
	info := &mesos.TaskInfo{
		Name:      proto.String(generateTaskName(&t.Pod)),
		TaskId:    mutil.NewTaskID(t.ID),
		Executor:  spec.Executor,
		Data:      spec.Data,
		Resources: spec.Resources,
		SlaveId:   mutil.NewSlaveID(spec.SlaveID),
	}

	return info, nil
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

func (t *T) Roles() []string {
	var roles []string

	if r, ok := t.Pod.ObjectMeta.Labels[meta.RolesKey]; ok {
		roles = strings.Split(r, ",")

		for i, r := range roles {
			roles[i] = strings.TrimSpace(r)
		}

		roles = filterRoles(roles, not(emptyRole), not(seenRole()))
	} else {
		// no roles label defined,
		// by convention return the first allowed role
		// to be used for launching the pod task
		return []string{t.allowedRoles[0]}
	}

	return filterRoles(roles, inRoles(t.allowedRoles...))
}

func New(ctx api.Context, id string, pod *api.Pod, prototype *mesos.ExecutorInfo, allowedRoles []string) (*T, error) {
	if prototype == nil {
		return nil, fmt.Errorf("illegal argument: executor is nil")
	}

	if len(allowedRoles) == 0 {
		allowedRoles = defaultRoles
	}

	key, err := MakePodKey(ctx, pod.Name)
	if err != nil {
		return nil, err
	}

	if id == "" {
		id = "pod." + uuid.NewUUID().String()
	}

	task := &T{
		ID:           id,
		Pod:          *pod,
		State:        StatePending,
		podKey:       key,
		mapper:       NewHostPortMapper(pod),
		Flags:        make(map[FlagType]struct{}),
		prototype:    prototype,
		allowedRoles: allowedRoles,
	}
	task.CreateTime = time.Now()

	return task, nil
}

func (t *T) SaveRecoveryInfo(dict map[string]string, spec *Spec) {
	dict[meta.TaskIdKey] = t.ID
	dict[meta.SlaveIdKey] = spec.SlaveID
	dict[meta.ExecutorIdKey] = spec.Executor.ExecutorId.GetValue()
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
func RecoverFrom(pod api.Pod) (*T, error) {
	// we cannot recover pods which are not bound because they don't have annotations yet
	// Mesos will send us a status update if the task is still launching.
	if pod.Spec.NodeName == "" {
		log.V(1).Infof("skipping recovery for unbound pod %v/%v", pod.Namespace, pod.Name)
		return nil, nil
	}

	// only process pods that are not in a terminal state
	switch pod.Status.Phase {
	case api.PodPending, api.PodRunning, api.PodUnknown: // continue
	default:
		log.V(1).Infof("skipping recovery for terminal pod %v/%v", pod.Namespace, pod.Name)
		return nil, nil
	}

	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)
	key, err := MakePodKey(ctx, pod.Name)
	if err != nil {
		return nil, err
	}

	//TODO(jdef) recover ports (and other resource requirements?) from the pod spec as well

	now := time.Now()
	t := &T{
		Pod:        pod,
		CreateTime: now,
		podKey:     key,
		State:      StatePending, // possibly running? mesos will tell us during reconciliation
		Flags:      make(map[FlagType]struct{}),
		mapper:     NewHostPortMapper(&pod),
		launchTime: now,
		bindTime:   now,
	}

	if pod.Annotations[meta.BindingHostKey] == "" {
		return nil, fmt.Errorf("incomplete metadata: missing %v annotation. Task looks bound, but not launched.", meta.BindingHostKey)
	}

	t.ID = pod.Annotations[meta.TaskIdKey]
	if t.ID == "" {
		return nil, fmt.Errorf("incomplete metadata: missing %v annotation", meta.TaskIdKey)
	}

	t.Set(Launched)
	t.Set(Bound)

	return t, nil
}
