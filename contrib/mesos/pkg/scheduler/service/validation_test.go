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
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	mresource "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resource"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

func TestStaticPodValidator(t *testing.T) {
	// test within limits
	tests := []struct {
		// given
		pods <-chan *api.Pod
		// wants
		podcount int
		cputot   float64
		memtot   float64
	}{
		// test: valid, pod specifies limits for ALL containers
		{
			pods: _pods(_pod(
				_podName("foo", "bar"),
				_containers(
					_container(_resourceLimits(10, 20)),
					_container(_resourceLimits(30, 40)),
					_container(_resourceLimits(50, 60)),
				),
			)),
			podcount: 1,
			cputot:   90,
			memtot:   120,
		},
		// test: valid, multiple pods, specify limits for ALL containers
		{
			pods: _pods(
				_pod(
					_podName("foo", "bar"),
					_containers(
						_container(_resourceLimits(10, 20)),
						_container(_resourceLimits(30, 40)),
						_container(_resourceLimits(50, 60)),
					),
				),
				_pod(
					_podName("kjh", "jkk"),
					_containers(
						_container(_resourceLimits(15, 25)),
						_container(_resourceLimits(35, 45)),
						_container(_resourceLimits(55, 65)),
					),
				),
			),
			podcount: 2,
			cputot:   195,
			memtot:   255,
		},
		// test: no limits on CT in first pod so it's rejected
		{
			pods: _pods(
				_pod(
					_podName("foo", "bar"),
					_containers(
						_container(_resourceLimits(10, 20)),
						_container(), // results in pod rejection
						_container(_resourceLimits(50, 60)),
					),
				),
				_pod(
					_podName("wza", "wer"),
					_containers(
						_container(_resourceLimits(10, 20)),
						_container(_resourceLimits(30, 40)),
						_container(_resourceLimits(50, 60)),
					),
				),
			),
			podcount: 1,
			cputot:   90,
			memtot:   120,
		},
	}
	for i, tc := range tests {
		var cpu, mem float64
		f := staticPodValidator(&cpu, &mem, false)
		list := podutil.List(f.Do(tc.pods))
		assert.Equal(t, tc.podcount, len(list.Items), "test case #%d: expected %d pods instead of %d", i, tc.podcount, len(list.Items))
		assert.EqualValues(t, tc.cputot, cpu, "test case #%d: expected %f total cpu instead of %f", i, tc.cputot, cpu)
		assert.EqualValues(t, tc.memtot, mem, "test case #%d: expected %f total mem instead of %f", i, tc.memtot, mem)
	}
}

type podOpt func(*api.Pod)
type ctOpt func(*api.Container)

func _pods(pods ...*api.Pod) <-chan *api.Pod {
	ch := make(chan *api.Pod, len(pods))
	for _, x := range pods {
		ch <- x
	}
	close(ch)
	return ch
}

func _pod(opts ...podOpt) *api.Pod {
	p := &api.Pod{}
	for _, x := range opts {
		x(p)
	}
	return p
}

func _container(opts ...ctOpt) (c api.Container) {
	for _, x := range opts {
		x(&c)
	}
	return
}

func _containers(ct ...api.Container) podOpt {
	return podOpt(func(p *api.Pod) {
		p.Spec.Containers = ct
	})
}

func _resourceLimits(cpu mresource.CPUShares, mem mresource.MegaBytes) ctOpt {
	return ctOpt(func(c *api.Container) {
		if c.Resources.Limits == nil {
			c.Resources.Limits = make(api.ResourceList)
		}
		c.Resources.Limits[api.ResourceCPU] = *resource.NewMilliQuantity(int64(float64(cpu)*1000.0), resource.DecimalSI)
		c.Resources.Limits[api.ResourceMemory] = *resource.NewQuantity(int64(float64(mem)*1024.0*1024.0), resource.BinarySI)
	})
}

func _podName(ns, name string) podOpt {
	return podOpt(func(p *api.Pod) {
		p.Namespace = ns
		p.Name = name
	})
}
