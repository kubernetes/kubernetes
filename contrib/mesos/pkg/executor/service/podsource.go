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

package service

import (
	"k8s.io/kubernetes/contrib/mesos/pkg/executor"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"

	log "github.com/golang/glog"
)

const (
	// if we don't use this source then the kubelet will do funny, mirror things. we alias
	// this here for convenience. see the docs for sourceMesos for additional explanation.
	// @see ConfigSourceAnnotationKey
	mesosSource = kubetypes.ApiserverSource
)

type (
	podName struct {
		namespace, name string
	}

	sourceMesos struct {
		stop          <-chan struct{}
		out           chan<- interface{} // never close this because pkg/util/config.mux doesn't handle that very well
		registry      executor.Registry
		priorPodNames map[podName]string // map podName to taskID
	}
)

func newSourceMesos(
	stop <-chan struct{},
	out chan<- interface{},
	podWatch *cache.ListWatch,
	registry executor.Registry,
) {
	source := &sourceMesos{
		stop:          stop,
		out:           out,
		registry:      registry,
		priorPodNames: make(map[podName]string),
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

// send is an update callback invoked by NewUndeltaStore
func (source *sourceMesos) send(objs []interface{}) {
	var (
		pods     = make([]*api.Pod, 0, len(objs))
		podNames = make(map[podName]string, len(objs))
	)

	for _, o := range objs {
		p := o.(*api.Pod)
		addPod := false
		if _, ok := p.Annotations[kubetypes.ConfigMirrorAnnotationKey]; ok {
			// pass through all mirror pods
			addPod = true
		} else if rpod, err := source.registry.Update(p); err == nil {
			// pod is bound to a task, and the update is compatible
			// so we'll allow it through
			addPod = true
			p = rpod.Pod() // use the (possibly) updated pod spec!
			podNames[podName{p.Namespace, p.Name}] = rpod.Task()
		} else if rpod != nil {
			// we were able to ID the pod but the update still failed...
			log.Warningf("failed to update registry for task %v pod %v/%v: %v",
				rpod.Task(), p.Namespace, p.Name, err)
		} else {
			// unrecognized pod, skip!
			log.V(2).Infof("skipping pod %v/%v", p.Namespace, p.Name)
		}

		if addPod {
			pods = append(pods, p)
		}
	}

	// detect when pods are deleted and notify the registry
	for k, taskID := range source.priorPodNames {
		if _, found := podNames[k]; !found {
			source.registry.Remove(taskID)
		}
	}

	source.priorPodNames = podNames

	u := kubetypes.PodUpdate{
		Op:     kubetypes.SET,
		Pods:   pods,
		Source: mesosSource,
	}
	select {
	case <-source.stop:
	default:
		select {
		case <-source.stop:
		case source.out <- u:
		}
	}
	log.V(2).Infof("sent %d pod updates", len(pods))
}
