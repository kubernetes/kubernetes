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

package e2e

import (
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
)

const (
	consumptionTimeInSeconds = 30
	sleepTime                = 30 * time.Second
	requestSizeInMillicores  = 100
	port                     = 80
	targetPort               = 8080
	timeoutRC                = 120 * time.Second
	image                    = "gcr.io/google_containers/resource_consumer:alpha"
	rcIsNil                  = "ERROR: replicationController = nil"
)

/*
ResourceConsumer is a tool for testing. It helps create specified usage of CPU or memory (Warnig: memory not supported)
typical use case:
rc.ConsumeCPU(600)
// ... check your assumption here
rc.ConsumeCPU(300)
// ... check your assumption here
*/
type ResourceConsumer struct {
	name      string
	framework *Framework
	channel   chan int
	stop      chan int
}

// NewResourceConsumer creates new ResourceConsumer
// cpu argument is in millicores
func NewResourceConsumer(name string, replicas int, cpu int, framework *Framework) *ResourceConsumer {
	runServiceAndRCForResourceConsumer(framework.Client, framework.Namespace.Name, name, replicas)
	rc := &ResourceConsumer{
		name:      name,
		framework: framework,
		channel:   make(chan int),
		stop:      make(chan int),
	}
	go rc.makeConsumeCPURequests()
	rc.ConsumeCPU(cpu)
	return rc
}

// ConsumeCPU consumes given number of CPU
func (rc *ResourceConsumer) ConsumeCPU(millicores int) {
	rc.channel <- millicores
}

func (rc *ResourceConsumer) makeConsumeCPURequests() {
	defer GinkgoRecover()
	var count int
	var rest int
	for {
		select {
		case millicores := <-rc.channel:
			count = millicores / requestSizeInMillicores
			rest = millicores - count*requestSizeInMillicores
		case <-time.After(sleepTime):
			if count > 0 {
				rc.sendConsumeCPUrequests(count, requestSizeInMillicores, consumptionTimeInSeconds)
			}
			if rest > 0 {
				go rc.sendOneConsumeCPUrequest(rest, consumptionTimeInSeconds)
			}
		case <-rc.stop:
			return
		}
	}
}

func (rc *ResourceConsumer) sendConsumeCPUrequests(requests, millicores, durationSec int) {
	for i := 0; i < requests; i++ {
		go rc.sendOneConsumeCPUrequest(millicores, durationSec)
	}
}

// sendOneConsumeCPUrequest sends POST request for cpu consumption
func (rc *ResourceConsumer) sendOneConsumeCPUrequest(millicores int, durationSec int) {
	_, err := rc.framework.Client.Post().
		Prefix("proxy").
		Namespace(rc.framework.Namespace.Name).
		Resource("services").
		Name(rc.name).
		Suffix("ConsumeCPU").
		Param("millicores", strconv.Itoa(millicores)).
		Param("durationSec", strconv.Itoa(durationSec)).
		Do().
		Raw()
	expectNoError(err)
}

func (rc *ResourceConsumer) GetReplicas() int {
	replicationController, err := rc.framework.Client.ReplicationControllers(rc.framework.Namespace.Name).Get(rc.name)
	expectNoError(err)
	if replicationController == nil {
		Failf(rcIsNil)
	}
	return replicationController.Status.Replicas
}

func (rc *ResourceConsumer) WaitForReplicas(desiredReplicas int) {
	timeout := 10 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		if desiredReplicas == rc.GetReplicas() {
			Logf("Replication Controller current replicas number is equal to desired replicas number: %d", desiredReplicas)
			return
		} else {
			Logf("Replication Controller current replicas number %d waiting to be %d", rc.GetReplicas(), desiredReplicas)
		}
	}
	Failf("timeout waiting %v for pods size to be %d", timeout, desiredReplicas)
}

func (rc *ResourceConsumer) CleanUp() {
	rc.stop <- 0
	expectNoError(DeleteRC(rc.framework.Client, rc.framework.Namespace.Name, rc.name))
	expectNoError(rc.framework.Client.Services(rc.framework.Namespace.Name).Delete(rc.name))
	expectNoError(rc.framework.Client.Experimental().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Delete(rc.name, api.NewDeleteOptions(0)))
}

func runServiceAndRCForResourceConsumer(c *client.Client, ns, name string, replicas int) {
	_, err := c.Services(ns).Create(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{
				Port:       port,
				TargetPort: util.NewIntOrStringFromInt(targetPort),
			}},

			Selector: map[string]string{
				"name": name,
			},
		},
	})
	expectNoError(err)
	config := RCConfig{
		Client:    c,
		Image:     image,
		Name:      name,
		Namespace: ns,
		Timeout:   timeoutRC,
		Replicas:  replicas,
	}
	expectNoError(RunRC(config))
}
