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

package service_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/service"
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
			pods: pods(pod(
				podName("foo", "bar"),
				containers(
					container(resourceLimits(10, 20)), // min is 32
					container(resourceLimits(30, 40)),
					container(resourceLimits(50, 60)),
				),
			)),
			podcount: 1,
			cputot:   90,
			memtot:   132,
		},
		// test: valid, multiple pods, specify limits for ALL containers
		{
			pods: pods(
				pod(
					podName("foo", "bar"),
					containers(
						container(resourceLimits(10, 20)), // min is 32
						container(resourceLimits(30, 40)),
						container(resourceLimits(50, 60)),
					),
				),
				pod(
					podName("kjh", "jkk"),
					containers(
						container(resourceLimits(15, 25)), // min is 32
						container(resourceLimits(35, 45)),
						container(resourceLimits(55, 65)),
					),
				),
			),
			podcount: 2,
			cputot:   195,
			memtot:   274,
		},
		// test: no limits on CT in first pod so it's rejected
		{
			pods: pods(
				pod(
					podName("foo", "bar"),
					containers(
						container(resourceLimits(10, 20)), // min is 32
						container(),                       // min is 0.01, 32
						container(resourceLimits(50, 60)),
					),
				),
				pod(
					podName("wza", "wer"),
					containers(
						container(resourceLimits(10, 20)), // min is 32
						container(resourceLimits(30, 40)),
						container(resourceLimits(50, 60)),
					),
				),
			),
			podcount: 2,
			cputot:   60.01 + 90,
			memtot:   124 + 132,
		},
	}
	for i, tc := range tests {
		var cpu, mem float64
		f := service.StaticPodValidator(0, 0, &cpu, &mem)
		list := podutil.List(f.Do(tc.pods))
		assert.Equal(t, tc.podcount, len(list.Items), "test case #%d: expected %d pods instead of %d", i, tc.podcount, len(list.Items))
		assert.EqualValues(t, tc.cputot, cpu, "test case #%d: expected %f total cpu instead of %f", i, tc.cputot, cpu)
		assert.EqualValues(t, tc.memtot, mem, "test case #%d: expected %f total mem instead of %f", i, tc.memtot, mem)
	}
}

type podOpt func(*api.Pod)
type ctOpt func(*api.Container)

func pods(pods ...*api.Pod) <-chan *api.Pod {
	ch := make(chan *api.Pod, len(pods))
	for _, x := range pods {
		ch <- x
	}
	close(ch)
	return ch
}

func pod(opts ...podOpt) *api.Pod {
	p := &api.Pod{}
	for _, x := range opts {
		x(p)
	}
	return p
}

func container(opts ...ctOpt) (c api.Container) {
	for _, x := range opts {
		x(&c)
	}
	return
}

func containers(ct ...api.Container) podOpt {
	return podOpt(func(p *api.Pod) {
		p.Spec.Containers = ct
	})
}

func resourceLimits(cpu resources.CPUShares, mem resources.MegaBytes) ctOpt {
	return ctOpt(func(c *api.Container) {
		if c.Resources.Limits == nil {
			c.Resources.Limits = make(api.ResourceList)
		}
		c.Resources.Limits[api.ResourceCPU] = *resource.NewMilliQuantity(int64(float64(cpu)*1000.0), resource.DecimalSI)
		c.Resources.Limits[api.ResourceMemory] = *resource.NewQuantity(int64(float64(mem)*1024.0*1024.0), resource.BinarySI)
	})
}

func podName(ns, name string) podOpt {
	return podOpt(func(p *api.Pod) {
		p.Namespace = ns
		p.Name = name
	})
}
