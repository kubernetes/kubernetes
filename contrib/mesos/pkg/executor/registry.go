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

package executor

import (
	"encoding/json"
	"errors"
	"sync"

	"k8s.io/kubernetes/contrib/mesos/pkg/executor/messages"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	log "github.com/golang/glog"
)

type (
	RegistrationEvent int

	RegisteredPod struct {
		*api.Pod
		taskID string
		event  RegistrationEvent
	}

	Registry interface {
		Update(pod *api.Pod) (*RegisteredPod, error)
		Remove(taskID string) error

		bind(taskID string, pod *api.Pod) error
		// watch the registered pod event stream, returns nil if the watch fails
		watch() <-chan *RegisteredPod
		empty() bool
		pod(taskID string) *api.Pod
		shutdown()
	}

	registryImpl struct {
		client     *client.Client
		b          *broadcast
		updates    chan<- *RegisteredPod
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
	RegistrationEventBound RegistrationEvent = iota
	RegistrationEventUpdated
	RegistrationEventDeleted
	RegistrationEventIncompatibleUpdate
)

func IsUnsupportedUpdate(err error) bool {
	return err == errUnsupportedUpdate
}

func (rp *RegisteredPod) Task() string {
	return rp.taskID
}

func NewRegistry(client *client.Client) Registry {
	updates := make(chan *RegisteredPod)
	r := &registryImpl{
		client:     client,
		b:          newBroadcast(updates, 0),
		updates:    updates,
		boundTasks: make(map[string]*api.Pod),
	}
	go r.b.run()
	return r
}

func (r registryImpl) watch() <-chan *RegisteredPod {
	x, err := r.b.listen()
	if err != nil {
		log.Error(err)
		return nil
	}
	return x.in
}

func taskIDFor(pod *api.Pod) (taskID string, err error) {
	taskID = pod.Annotations[meta.TaskIdKey]
	if taskID == "" {
		err = errUnknownTask
	}
	return
}

func (r registryImpl) shutdown() {
	//TODO(jdef) flesh this out
	r.m.Lock()
	defer r.m.Unlock()
	r.boundTasks = map[string]*api.Pod{}
}

func (r registryImpl) empty() bool {
	r.m.RLock()
	defer r.m.RUnlock()
	return len(r.boundTasks) == 0
}

func (r registryImpl) pod(taskID string) *api.Pod {
	r.m.RLock()
	defer r.m.RUnlock()
	return r.boundTasks[taskID]
}

func (r registryImpl) Remove(taskID string) error {
	r.m.Lock()
	defer r.m.Unlock()
	pod, ok := r.boundTasks[taskID]
	if !ok {
		return errUnknownTask
	}

	delete(r.boundTasks, taskID)

	r.updates <- &RegisteredPod{
		Pod:    pod,
		taskID: taskID,
		event:  RegistrationEventDeleted,
	}

	log.V(1).Infof("unbound task %v from pod %v/%v", taskID, pod.Namespace, pod.Name)
	return nil
}

func (r registryImpl) Update(pod *api.Pod) (*RegisteredPod, error) {
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

	r.m.Lock()
	defer r.m.Unlock()
	oldPod, ok := r.boundTasks[taskID]
	if !ok {
		return nil, errUnknownTask
	}

	registeredPod := &RegisteredPod{
		Pod:    pod,
		taskID: taskID,
		event:  RegistrationEventUpdated,
	}

	// TODO(jdef) would be nice to only execute this logic based on the presence of
	// some particular annotation.
	//   - preserve the original container port spec since the k8sm scheduler
	//   has likely changed it.
	containers := oldPod.Spec.Containers
	ctPorts := make(map[string][]api.ContainerPort, len(containers))
	for i := range containers {
		ctPorts[containers[i].Name] = containers[i].Ports
	}
	containers = pod.Spec.Containers
	for i := range containers {
		name := containers[i].Name
		if ports, found := ctPorts[name]; found {
			containers[i].Ports = ports
			delete(ctPorts, name)
		} else {
			// old pod spec is missing this container?!
			goto incompatibleUpdate
		}
	}
	if len(ctPorts) > 0 {
		// new pod spec has containers that aren't in the old pod spec
		goto incompatibleUpdate
	}

	// update our internal copy and broadcast the change
	r.boundTasks[taskID] = pod
	r.updates <- registeredPod

	log.V(1).Infof("updated task %v pod %v/%v", taskID, pod.Namespace, pod.Name)
	return registeredPod, nil

incompatibleUpdate:
	registeredPod.event = RegistrationEventIncompatibleUpdate
	r.updates <- registeredPod

	log.Warningf("pod containers changed in an incompatible way; aborting update")
	return registeredPod, errUnsupportedUpdate
}

func (r registryImpl) bind(taskID string, pod *api.Pod) error {

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

	// TODO(k8s): use Pods interface for binding once clusters are upgraded
	// return b.Pods(binding.Namespace).Bind(binding)
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
		log.Infof("Binding '%v/%v' to '%v' with annotations %+v...", pod.Namespace, pod.Name, binding.Target.Name, binding.Annotations)
		ctx := api.WithNamespace(api.NewContext(), binding.Namespace)
		err := r.client.Post().Namespace(api.NamespaceValue(ctx)).Resource("bindings").Body(binding).Do().Error()
		if err != nil {
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
		log.V(4).Infof("Patching annotations %v of pod %v/%v: %v", pod.Annotations, pod.Namespace, pod.Name, string(patchJson))
		err := r.client.Patch(api.MergePatchType).RequestURI(pod.SelfLink).Body(patchJson).Do().Error()
		if err != nil {
			log.Errorf("Error updating annotations of ready-to-launch pod %v/%v: %v", pod.Namespace, pod.Name, err)
			return errAnnotationUpdateFailure
		}
	}

	boundSuccessfully = true

	r.updates <- &RegisteredPod{
		Pod:    pod,
		taskID: taskID,
		event:  RegistrationEventBound,
	}

	log.V(1).Infof("bound task %v to pod %v/%v", taskID, pod.Namespace, pod.Name)
	return nil
}
