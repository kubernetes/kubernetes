/*
Copyright 2015 The Kubernetes Authors.

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

package executor

import (
	"encoding/json"
	"errors"
	"sync"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"k8s.io/kubernetes/contrib/mesos/pkg/executor/messages"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"

	log "github.com/golang/glog"
)

type (
	podEventType int

	PodEvent struct {
		pod       *api.Pod
		taskID    string
		eventType podEventType
	}

	// Registry is a state store for pod task metadata. Clients are expected to watch() the
	// event stream to observe changes over time.
	Registry interface {
		// Update modifies the registry's iternal representation of the pod; it may also
		// modify the pod argument itself. An update may fail because either a pod isn't
		// labeled with a task ID, the task ID is unknown, or the nature of the update may
		// be incompatible with what's supported in kubernetes-mesos.
		Update(pod *api.Pod) (*PodEvent, error)

		// Remove the task from this registry, returns an error if the taskID is unknown.
		Remove(taskID string) error

		// bind associates a taskID with a pod, triggers the binding API on the k8s apiserver
		// and stores the resulting pod-task metadata.
		bind(taskID string, pod *api.Pod) error

		// watch returns the event stream of the registry. clients are expected to read this
		// stream otherwise the event buffer will fill up and registry ops will block.
		watch() <-chan *PodEvent

		// return true if there are no tasks registered
		empty() bool

		// return the api.Pod registered to the given taskID or else nil
		pod(taskID string) *api.Pod

		// shutdown any related async processing and clear the internal state of the registry
		shutdown()
	}

	registryImpl struct {
		client     *clientset.Clientset
		updates    chan *PodEvent
		m          sync.RWMutex
		boundTasks map[string]*api.Pod
	}
)

var (
	errCreateBindingFailed     = errors.New(messages.CreateBindingFailure)
	errAnnotationUpdateFailure = errors.New(messages.AnnotationUpdateFailure)
	errUnknownTask             = errors.New("unknown task ID")
	errUnsupportedUpdate       = errors.New("pod update allowed by k8s is incompatible with this version of k8s-mesos")
)

const (
	PodEventBound podEventType = iota
	PodEventUpdated
	PodEventDeleted
	PodEventIncompatibleUpdate

	updatesBacklogSize = 200
)

func IsUnsupportedUpdate(err error) bool {
	return err == errUnsupportedUpdate
}

func (rp *PodEvent) Task() string {
	return rp.taskID
}

func (rp *PodEvent) Pod() *api.Pod {
	return rp.pod
}

func (rp *PodEvent) FormatShort() string {
	return "task '" + rp.taskID + "' pod '" + rp.pod.Namespace + "/" + rp.pod.Name + "'"
}

func NewRegistry(client *clientset.Clientset) Registry {
	r := &registryImpl{
		client:     client,
		updates:    make(chan *PodEvent, updatesBacklogSize),
		boundTasks: make(map[string]*api.Pod),
	}
	return r
}

func (r *registryImpl) watch() <-chan *PodEvent {
	return r.updates
}

func taskIDFor(pod *api.Pod) (taskID string, err error) {
	taskID = pod.Annotations[meta.TaskIdKey]
	if taskID == "" {
		err = errUnknownTask
	}
	return
}

func (r *registryImpl) shutdown() {
	//TODO(jdef) flesh this out
	r.m.Lock()
	defer r.m.Unlock()
	r.boundTasks = map[string]*api.Pod{}
}

func (r *registryImpl) empty() bool {
	r.m.RLock()
	defer r.m.RUnlock()
	return len(r.boundTasks) == 0
}

func (r *registryImpl) pod(taskID string) *api.Pod {
	r.m.RLock()
	defer r.m.RUnlock()
	return r.boundTasks[taskID]
}

func (r *registryImpl) Remove(taskID string) error {
	r.m.Lock()
	defer r.m.Unlock()
	pod, ok := r.boundTasks[taskID]
	if !ok {
		return errUnknownTask
	}

	delete(r.boundTasks, taskID)

	r.updates <- &PodEvent{
		pod:       pod,
		taskID:    taskID,
		eventType: PodEventDeleted,
	}

	log.V(1).Infof("unbound task %v from pod %v/%v", taskID, pod.Namespace, pod.Name)
	return nil
}

func (r *registryImpl) Update(pod *api.Pod) (*PodEvent, error) {
	// Don't do anything for pods without task anotation which means:
	// - "pre-scheduled" pods which have a NodeName set to this node without being scheduled already.
	// - static/mirror pods: they'll never have a TaskID annotation, and we don't expect them to ever change.
	// - all other pods that haven't passed through the launch-task-binding phase, which would set annotations.
	taskID, err := taskIDFor(pod)
	if err != nil {
		// There also could be a race between the overall launch-task process and this update, but here we
		// will never be able to process such a stale update because the "update pod" that we're receiving
		// in this func won't yet have a task ID annotation. It follows that we can safely drop such a stale
		// update on the floor because we'll get another update later that, in addition to the changes that
		// we're dropping now, will also include the changes from the binding process.
		log.V(5).Infof("ignoring pod update for %s/%s because %s annotation is missing", pod.Namespace, pod.Name, meta.TaskIdKey)
		return nil, err
	}

	// be a good citizen: copy the arg before making any changes to it
	clone, err := api.Scheme.DeepCopy(pod)
	if err != nil {
		return nil, err
	}
	pod = clone.(*api.Pod)

	r.m.Lock()
	defer r.m.Unlock()
	oldPod, ok := r.boundTasks[taskID]
	if !ok {
		return nil, errUnknownTask
	}

	registeredPod := &PodEvent{
		pod:       pod,
		taskID:    taskID,
		eventType: PodEventUpdated,
	}

	// TODO(jdef) would be nice to only execute this logic based on the presence of
	// some particular annotation:
	//   - preserve the original container port spec since the k8sm scheduler
	//     has likely changed it.
	if !copyPorts(pod, oldPod) {
		// TODO(jdef) the state of "pod" is possibly inconsistent at this point.
		// we don't care for the moment - we might later.
		registeredPod.eventType = PodEventIncompatibleUpdate
		r.updates <- registeredPod
		log.Warningf("pod containers changed in an incompatible way; aborting update")
		return registeredPod, errUnsupportedUpdate
	}

	// update our internal copy and broadcast the change
	r.boundTasks[taskID] = pod
	r.updates <- registeredPod

	log.V(1).Infof("updated task %v pod %v/%v", taskID, pod.Namespace, pod.Name)
	return registeredPod, nil
}

// copyPorts copies the container pod specs from src to dest and returns
// true if all ports (in both dest and src) are accounted for, otherwise
// false. if returning false then it's possible that only a partial copy
// has been performed.
func copyPorts(dest, src *api.Pod) bool {
	containers := src.Spec.Containers
	ctPorts := make(map[string][]api.ContainerPort, len(containers))
	for i := range containers {
		ctPorts[containers[i].Name] = containers[i].Ports
	}
	containers = dest.Spec.Containers
	for i := range containers {
		name := containers[i].Name
		if ports, found := ctPorts[name]; found {
			containers[i].Ports = ports
			delete(ctPorts, name)
		} else {
			// old pod spec is missing this container?!
			return false
		}
	}
	if len(ctPorts) > 0 {
		// new pod spec has containers that aren't in the old pod spec
		return false
	}
	return true
}

func (r *registryImpl) bind(taskID string, pod *api.Pod) error {
	// validate taskID matches that of the annotation
	annotatedTaskID, err := taskIDFor(pod)
	if err != nil {
		log.Warning("failed to bind: missing task ID annotation for pod ", pod.Namespace+"/"+pod.Name)
		return errCreateBindingFailed
	}
	if annotatedTaskID != taskID {
		log.Warningf("failed to bind: expected task-id %v instead of %v for pod %v/%v", taskID, annotatedTaskID, pod.Namespace, pod.Name)
		return errCreateBindingFailed
	}

	// record this as a bound task for now so that we can avoid racing with the mesos pod source, who is
	// watching the apiserver for pod updates and will verify pod-task validity with us upon receiving such
	boundSuccessfully := false
	defer func() {
		if !boundSuccessfully {
			r.m.Lock()
			defer r.m.Unlock()
			delete(r.boundTasks, taskID)
		}
	}()
	func() {
		r.m.Lock()
		defer r.m.Unlock()
		r.boundTasks[taskID] = pod
	}()

	if pod.Spec.NodeName == "" {
		//HACK(jdef): cloned binding construction from k8s plugin/pkg/scheduler/framework.go
		binding := &api.Binding{
			ObjectMeta: api.ObjectMeta{
				Namespace:   pod.Namespace,
				Name:        pod.Name,
				Annotations: make(map[string]string),
			},
			Target: api.ObjectReference{
				Kind: "Node",
				Name: pod.Annotations[meta.BindingHostKey],
			},
		}

		// forward the annotations that the scheduler wants to apply
		for k, v := range pod.Annotations {
			binding.Annotations[k] = v
		}

		// create binding on apiserver
		log.Infof("Binding task %v pod '%v/%v' to '%v' with annotations %+v...",
			taskID, pod.Namespace, pod.Name, binding.Target.Name, binding.Annotations)
		ctx := api.WithNamespace(api.NewContext(), binding.Namespace)
		err := r.client.CoreClient.Post().Namespace(api.NamespaceValue(ctx)).Resource("bindings").Body(binding).Do().Error()
		if err != nil {
			log.Warningf("failed to bind task %v pod %v/%v: %v", taskID, pod.Namespace, pod.Name, err)
			return errCreateBindingFailed
		}
	} else {
		// post annotations update to apiserver
		patch := struct {
			Metadata struct {
				Annotations map[string]string `json:"annotations"`
			} `json:"metadata"`
		}{}
		patch.Metadata.Annotations = pod.Annotations
		patchJson, _ := json.Marshal(patch)
		log.V(4).Infof("Patching annotations %v of task %v pod %v/%v: %v", pod.Annotations, taskID, pod.Namespace, pod.Name, string(patchJson))
		err := r.client.CoreClient.Patch(api.MergePatchType).RequestURI(pod.SelfLink).Body(patchJson).Do().Error()
		if err != nil {
			log.Errorf("Error updating annotations of ready-to-launch task %v pod %v/%v: %v", taskID, pod.Namespace, pod.Name, err)
			return errAnnotationUpdateFailure
		}
	}

	boundSuccessfully = true

	r.updates <- &PodEvent{
		pod:       pod,
		taskID:    taskID,
		eventType: PodEventBound,
	}

	log.V(1).Infof("bound task %v to pod %v/%v", taskID, pod.Namespace, pod.Name)
	return nil
}
