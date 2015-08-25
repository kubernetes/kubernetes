/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package componentcontroller

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/component"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"

	"github.com/golang/glog"
)

const (
	ComponentProbePeriod = 5 * time.Second
)

// NewEndpointController returns a new *EndpointController.
func NewComponentController(client *client.Client) *ComponentController {
	return &ComponentController{
		client:         client,
		componentStore: component.NewCache(),
		queue:          workqueue.New(),
		prober:         component.NewProber(),
	}
}

// EndpointController manages selector-based service endpoints.
type ComponentController struct {
	client *client.Client

	// store and synchronize, instead of just caching
	componentStore *component.Cache

	// Components that need to be probed.
	queue *workqueue.Type

	prober component.Prober
}

// Runs e; will not return until stopCh is closed. workers determines how many
// endpoints will be handled in parallel.
func (e *ComponentController) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()
	go util.Until(e.watcher, ComponentProbePeriod, stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(e.worker, time.Second, stopCh)
	}
	<-stopCh
	e.queue.ShutDown()
}

func (e *ComponentController) watcher() {
	// TODO: watch instead of polling
	list, err := e.client.ComponentsClient().List(labels.Everything(), fields.Everything())
	if err != nil {
		glog.Warningf("Failed to list components: %v", err)
	}
	for _, component := range list.Items {
		// TODO: clean up storage to do proper syncing
		created := e.componentStore.Create(component.Name, component)
		if !created {
			cached, _ := e.componentStore.Read(component.Name)
			if cached.ResourceVersion != component.ResourceVersion {
				glog.Warningf(
					"Cached component %q is out of sync; Cached: %+v New: %+v",
					component.Name,
					cached,
					component,
				)
				//TODO: this will cause the next update to fail, unless ResourceVersion is zeroed out
			}
		}
		e.queue.Add(component.Name)
	}
}

// worker runs a worker thread that just dequeues items, processes them, and
// marks them done. You may run as many of these in parallel as you wish; the
// workqueue guarantees that they will not end up processing the same service
// at the same time.
func (e *ComponentController) worker() {
	for {
		func() {
			componentName, quit := e.queue.Get()
			if quit {
				return
			}
			// Use defer: in the unlikely event that there's a
			// panic, we'd still like this to get marked done--
			// otherwise the controller will not be able to probe
			// this component again until it is restarted.
			defer e.queue.Done(componentName)
			e.probeComponent(componentName.(string))
		}()
	}
}

func (e *ComponentController) probeComponent(componentName string) {
	component, found := e.componentStore.Read(componentName)
	if !found {
		//TODO: what does this mean?
		glog.Warningf("Component %q unknown - dequeued by worker: %q", componentName)
		return
	}

	conditionCh := make(chan api.ComponentCondition)
	conditions := make([]api.ComponentCondition, 0, 2)
	go func() {
		for {
			condition, open := <-conditionCh
			if !open {
				return
			}
			conditions = append(conditions, condition)
		}
	}()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		conditionCh <- e.probeLiveness(&component)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		conditionCh <- e.probeReadiness(&component)
	}()

	wg.Wait()
	close(conditionCh)

	component.Status.Conditions = conditions

	newComponent, err := e.client.ComponentsClient().UpdateStatus(&component)
	if err != nil {
		glog.Errorf("failed to update component %q status: %v", componentName, err)
	}
	updated := e.componentStore.Update(componentName, *newComponent)
	if !updated {
		glog.Warningf("Component %q not cached - skipping cache update", componentName)
		//TODO: does this mean the component record was removed form the cache while still in the queue?
	}
}

func (e *ComponentController) probeLiveness(c *api.Component) api.ComponentCondition {
	condition := api.ComponentCondition{
		Type: api.ComponentAlive,
	}

	result, err := e.prober.Probe(c, c.Spec.LivenessProbe)
	if err != nil {
		condition.Reason = "error"
		condition.Message = fmt.Sprintf("Liveness probe errored: %v", err)
	}

	switch result {
	case probe.Success:
		condition.Status = api.ConditionTrue
	case probe.Failure:
		condition.Status = api.ConditionFalse
	default:
		condition.Status = api.ConditionUnknown
	}

	glog.V(1).Infof("Component %q liveness condition: %s", c.Name, condition)
	return condition
}

func (e *ComponentController) probeReadiness(c *api.Component) api.ComponentCondition {
	condition := api.ComponentCondition{
		Type: api.ComponentReady,
	}

	result, err := e.prober.Probe(c, c.Spec.ReadinessProbe)
	if err != nil {
		condition.Reason = "error"
		condition.Message = fmt.Sprintf("Readiness probe errored: %v", err)
	}

	switch result {
	case probe.Success:
		condition.Status = api.ConditionTrue
	case probe.Failure:
		condition.Status = api.ConditionFalse
	default:
		condition.Status = api.ConditionUnknown
	}

	glog.V(1).Infof("Component %q readiness condition: %s", c.Name, condition)
	return condition
}
