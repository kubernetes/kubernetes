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

package podsource

import (
	"k8s.io/kubernetes/contrib/mesos/pkg/executor"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"

	log "github.com/golang/glog"
)

type (
	filterType int

	podName struct {
		namespace, name string
	}

	// Filter is invoked for each snapshot of pod state that passes through this source
	Filter interface {
		// Before is invoked before any pods are evaluated
		Before(podCount int)
		// Accept returns true if this pod should be accepted by the source; a value
		// of false results in the pod appearing to have been removed from apiserver.
		// If true, the caller should use the output pod value for the remainder of
		// the processing task. If false then the output pod value may be nil.
		Accept(*api.Pod) (*api.Pod, bool)
		// After is invoked after all pods have been evaluated
		After()
	}

	// FilterFunc is a simplified Filter implementation that only implements Filter.Accept, its
	// Before and After implementations are noop.
	FilterFunc func(*api.Pod) (*api.Pod, bool)

	Source struct {
		stop    <-chan struct{}
		out     chan<- interface{} // never close this because pkg/util/config.mux doesn't handle that very well
		filters []Filter           // additional filters to apply to pod objects
	}

	Option func(*Source)
)

const (
	// if we don't use this source then the kubelet will do funny, mirror things. we alias
	// this here for convenience. see the docs for Source for additional explanation.
	// @see ConfigSourceAnnotationKey
	MesosSource = kubetypes.ApiserverSource
)

func (f FilterFunc) Before(_ int)                         {}
func (f FilterFunc) After()                               {}
func (f FilterFunc) Accept(pod *api.Pod) (*api.Pod, bool) { return f(pod) }

// Mesos spawns a new pod source that watches API server for changes and collaborates with
// executor.Registry to generate api.Pod objects in a fashion that's very Mesos-aware.
func Mesos(
	stop <-chan struct{},
	out chan<- interface{},
	podWatch *cache.ListWatch,
	registry executor.Registry,
	options ...Option,
) {
	source := &Source{
		stop: stop,
		out:  out,
		filters: []Filter{
			FilterFunc(filterMirrorPod),
			&registeredPodFilter{registry: registry},
		},
	}
	// note: any filters added by options should be applied after the defaults
	for _, opt := range options {
		opt(source)
	}
	// reflect changes from the watch into a chan, filtered to include only mirror pods
	// (have an ConfigMirrorAnnotationKey attr)
	cache.NewReflector(
		podWatch,
		&api.Pod{},
		cache.NewUndeltaStore(source.send, cache.MetaNamespaceKeyFunc),
		0,
	).RunUntil(stop)
}

func filterMirrorPod(p *api.Pod) (*api.Pod, bool) {
	_, ok := (*p).Annotations[kubetypes.ConfigMirrorAnnotationKey]
	return p, ok
}

type registeredPodFilter struct {
	priorPodNames, podNames map[podName]string // maps a podName to a taskID
	registry                executor.Registry
}

func (rpf *registeredPodFilter) Before(podCount int) {
	rpf.priorPodNames = rpf.podNames
	rpf.podNames = make(map[podName]string, podCount)
}

func (rpf *registeredPodFilter) After() {
	// detect when pods are deleted and notify the registry
	for k, taskID := range rpf.priorPodNames {
		if _, found := rpf.podNames[k]; !found {
			rpf.registry.Remove(taskID)
		}
	}
}

func (rpf *registeredPodFilter) Accept(p *api.Pod) (*api.Pod, bool) {
	rpod, err := rpf.registry.Update(p)
	if err == nil {
		// pod is bound to a task, and the update is compatible
		// so we'll allow it through
		p = rpod.Pod() // use the (possibly) updated pod spec!
		rpf.podNames[podName{p.Namespace, p.Name}] = rpod.Task()
		return p, true
	}
	if rpod != nil {
		// we were able to ID the pod but the update still failed...
		log.Warningf("failed to update registry for task %v pod %v/%v: %v",
			rpod.Task(), p.Namespace, p.Name, err)
	}
	return nil, false
}

// send is an update callback invoked by NewUndeltaStore; it applies all of source.filters
// to the incoming pod snapshot and forwards a PodUpdate that contains a snapshot of all
// the pods that were accepted by the filters.
func (source *Source) send(objs []interface{}) {
	var (
		podCount = len(objs)
		pods     = make([]*api.Pod, 0, podCount)
	)

	for _, f := range source.filters {
		f.Before(podCount)
	}
foreachPod:
	for _, o := range objs {
		p := o.(*api.Pod)
		for _, f := range source.filters {
			if p, ok := f.Accept(p); ok {
				pods = append(pods, p)
				continue foreachPod
			}
		}
		// unrecognized pod
		log.V(2).Infof("skipping pod %v/%v", p.Namespace, p.Name)
	}
	// TODO(jdef) should these be applied in reverse order instead?
	for _, f := range source.filters {
		f.After()
	}

	u := kubetypes.PodUpdate{
		Op:     kubetypes.SET,
		Pods:   pods,
		Source: MesosSource,
	}
	select {
	case <-source.stop:
	case source.out <- u:
		log.V(2).Infof("sent %d pod updates", len(pods))
	}
}

func ContainerEnvOverlay(env []api.EnvVar) Option {
	return func(s *Source) {
		// prepend this filter so that it impacts *all* pods running on the slave
		s.filters = append([]Filter{filterContainerEnvOverlay(env)}, s.filters...)
	}
}

func filterContainerEnvOverlay(env []api.EnvVar) FilterFunc {
	f := podutil.Environment(env)
	return func(pod *api.Pod) (*api.Pod, bool) {
		f(pod)
		// we should't vote, let someone else decide whether the pod gets accepted
		return pod, false
	}
}
