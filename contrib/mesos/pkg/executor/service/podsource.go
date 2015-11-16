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
	"fmt"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	// if we don't use this source then the kubelet will do funny, mirror things. we alias
	// this here for convenience. see the docs for sourceMesos for additional explanation.
	// @see ConfigSourceAnnotationKey
	mesosSource = kubetypes.ApiserverSource
)

// sourceMesos merges pods from mesos, and mirror pods from the apiserver. why?
// (a) can't have two sources with the same name;
// (b) all sources, other than ApiserverSource are considered static/mirror
// sources, and;
// (c) kubelet wants to see mirror pods reflected in a non-static source.
//
// Mesos pods must appear to come from apiserver due to (b), while reflected
// static pods (mirror pods) must appear to come from apiserver due to (c).
//
// The only option I could think of was creating a source that merges the pod
// streams. I don't like it. But I could think of anything else, other than
// starting to hack up the kubelet's understanding of mirror/static pod
// sources (ouch!)
type sourceMesos struct {
	sourceFinished chan struct{}              // sourceFinished closes when mergeAndForward exits
	out            chan<- interface{}         // out is the sink for merged pod snapshots
	mirrorPods     chan []*api.Pod            // mirrorPods communicates snapshots of the current set of mirror pods
	execUpdates    <-chan kubetypes.PodUpdate // execUpdates receives snapshots of the current set of mesos pods
}

// newSourceMesos creates a pod config source that merges pod updates from
// mesos (via execUpdates), and mirror pod updates from the apiserver (via
// podWatch) writing the merged update stream to the out chan. It is expected
// that execUpdates will only ever contain SET operations. The source takes
// ownership of the sourceFinished chan, closing it when the source terminates.
// Source termination happens when the execUpdates chan is closed and fully
// drained of updates.
func newSourceMesos(
	sourceFinished chan struct{},
	execUpdates <-chan kubetypes.PodUpdate,
	out chan<- interface{},
	podWatch *cache.ListWatch,
) {
	source := &sourceMesos{
		sourceFinished: sourceFinished,
		mirrorPods:     make(chan []*api.Pod),
		execUpdates:    execUpdates,
		out:            out,
	}
	// reflect changes from the watch into a chan, filtered to include only mirror pods (have an ConfigMirrorAnnotationKey attr)
	cache.NewReflector(podWatch, &api.Pod{}, cache.NewUndeltaStore(source.send, cache.MetaNamespaceKeyFunc), 0).RunUntil(sourceFinished)
	go source.mergeAndForward()
}

func (source *sourceMesos) send(objs []interface{}) {
	var mirrors []*api.Pod
	for _, o := range objs {
		p := o.(*api.Pod)
		if _, ok := p.Annotations[kubetypes.ConfigMirrorAnnotationKey]; ok {
			mirrors = append(mirrors, p)
		}
	}
	select {
	case <-source.sourceFinished:
	case source.mirrorPods <- mirrors:
	}
}

func (source *sourceMesos) mergeAndForward() {
	// execUpdates will be closed by the executor on shutdown
	defer close(source.sourceFinished)
	var (
		mirrors = []*api.Pod{}
		pods    = []*api.Pod{}
	)
eventLoop:
	for {
		select {
		case m := <-source.mirrorPods:
			mirrors = m[:]
			u := kubetypes.PodUpdate{
				Op:     kubetypes.SET,
				Pods:   append(m, pods...),
				Source: mesosSource,
			}
			log.V(3).Infof("mirror update, sending snapshot of size %d", len(u.Pods))
			source.out <- u
		case u, ok := <-source.execUpdates:
			if !ok {
				break eventLoop
			}
			if u.Op != kubetypes.SET {
				panic(fmt.Sprintf("unexpected Op type: %v", u.Op))
			}

			pods = u.Pods[:]
			u.Pods = append(u.Pods, mirrors...)
			u.Source = mesosSource
			log.V(3).Infof("pods update, sending snapshot of size %d", len(u.Pods))
			source.out <- u
		}
	}
	log.V(2).Infoln("mesos pod source terminating normally")
}
