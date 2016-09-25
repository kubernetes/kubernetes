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

package queuer

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
)

// functional Pod option
type PodOpt func(*Pod)

// wrapper for the k8s pod type so that we can define additional methods on a "pod"
type Pod struct {
	*api.Pod
	deadline *time.Time
	delay    *time.Duration
	notify   queue.BreakChan
}

func NewPod(pod *api.Pod, opt ...PodOpt) *Pod {
	p := &Pod{Pod: pod}
	for _, f := range opt {
		f(p)
	}
	return p
}

// Deadline sets the deadline for a Pod
func Deadline(deadline time.Time) PodOpt {
	return func(pod *Pod) {
		pod.deadline = &deadline
	}
}

// Delay sets the delay for a Pod
func Delay(delay time.Duration) PodOpt {
	return func(pod *Pod) {
		pod.delay = &delay
	}
}

// Notify sets the breakout notification channel for a Pod
func Notify(notify queue.BreakChan) PodOpt {
	return func(pod *Pod) {
		pod.notify = notify
	}
}

// implements Copyable
func (p *Pod) Copy() queue.Copyable {
	if p == nil {
		return nil
	}
	//TODO(jdef) we may need a better "deep-copy" implementation
	pod := *(p.Pod)
	return &Pod{Pod: &pod}
}

// implements Unique
func (p *Pod) GetUID() string {
	if id, err := cache.MetaNamespaceKeyFunc(p.Pod); err != nil {
		panic(fmt.Sprintf("failed to determine pod id for '%+v'", p.Pod))
	} else {
		return id
	}
}

// implements Deadlined
func (dp *Pod) Deadline() (time.Time, bool) {
	if dp.deadline != nil {
		return *(dp.deadline), true
	}
	return time.Time{}, false
}

func (dp *Pod) GetDelay() time.Duration {
	if dp.delay != nil {
		return *(dp.delay)
	}
	return 0
}

func (p *Pod) Breaker() queue.BreakChan {
	return p.notify
}

func (p *Pod) String() string {
	displayDeadline := "<none>"
	if deadline, ok := p.Deadline(); ok {
		displayDeadline = deadline.String()
	}
	return fmt.Sprintf("{pod:%v, deadline:%v, delay:%v}", p.Pod.Name, displayDeadline, p.GetDelay())
}
