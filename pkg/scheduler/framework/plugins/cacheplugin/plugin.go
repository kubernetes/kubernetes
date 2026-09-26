/*
Copyright The Kubernetes Authors.

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

package cacheplugin

import (
	"context"
	"sync"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	listersv1 "k8s.io/client-go/listers/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

const Name = "CachePlugin"

// New initializes and returns a new Coscheduling plugin.
func New(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
	return cacheplugin, nil
}

var cacheplugin = &CachePlugin{
	caches: make([]CacheForPlugin, 0),
}

type CachePlugin struct {
	lock sync.Mutex

	caches []CacheForPlugin
}

var _ fwk.ReservePlugin = &CachePlugin{}

func (c *CachePlugin) Name() string {
	return Name
}

func (c *CachePlugin) Reserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	c.lock.Lock()
	defer c.lock.Unlock()
	for _, cache := range c.caches {
		cache.ProcessReservePod(p, nodeName)
	}
	return nil
}

func (c *CachePlugin) Unreserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
	c.lock.Lock()
	defer c.lock.Unlock()
	for _, cache := range c.caches {
		cache.ProcessUnreservePod(p, nodeName)
	}
}

func GetPodFromLister(key NamespaceedNameNode, pl listersv1.PodLister) (p *corev1.Pod, err error) {
	p, err = pl.Pods(key.Namespace).Get(key.Name)
	return
}
