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
	requestSizeInMilicores   = 100
	port                     = 80
	targetPort               = 8080
	timeoutRC                = 120 * time.Second
	image                    = "gcr.io/google_containers/resource_consumer:alpha"
)

/*
ConsumingRC is a tool for testing. It helps create specified usage of CPU or memory (Warnig: memory not supported)
typical use case:
rc.ConsumeCPU(600)
// ... check your assumption here
rc.ConsumeCPU(300)
// ... check your assumption here
*/
type ConsumingRC struct {
	name      string
	framework *Framework
	channel   chan int
	stop      chan int
}

// NewConsumingRC creates new ConsumingRC
func NewConsumingRC(name string, replicas int, framework *Framework) *ConsumingRC {
	startService(framework.Client, framework.Namespace.Name, name, replicas)
	rc := &ConsumingRC{
		name:      name,
		framework: framework,
		channel:   make(chan int),
		stop:      make(chan int),
	}
	go rc.makeConsumeCPURequests()
	rc.ConsumeCPU(0)
	return rc
}

// ConsumeCPU consumes given number of CPU
func (rc *ConsumingRC) ConsumeCPU(milicores int) {
	rc.channel <- milicores
}

func (rc *ConsumingRC) makeConsumeCPURequests() {
	defer GinkgoRecover()
	var count int
	var rest int
	for {
		select {
		case milicores := <-rc.channel:
			count = milicores / requestSizeInMilicores
			rest = milicores - count*requestSizeInMilicores
		case <-time.After(sleepTime):
			if count > 0 {
				rc.sendConsumeCPUrequests(count, requestSizeInMilicores, consumptionTimeInSeconds)
			}
			if rest > 0 {
				go rc.sendOneConsumeCPUrequest(rest, consumptionTimeInSeconds)
			}
		case <-rc.stop:
			return
		}
	}
}

func (rc *ConsumingRC) sendConsumeCPUrequests(requests, milicores, durationSec int) {
	for i := 0; i < requests; i++ {
		go rc.sendOneConsumeCPUrequest(milicores, durationSec)
	}
}

// sendOneConsumeCPUrequest sends POST request for cpu consumption
func (rc *ConsumingRC) sendOneConsumeCPUrequest(milicores int, durationSec int) {
	_, err := rc.framework.Client.Post().
		Prefix("proxy").
		Namespace(rc.framework.Namespace.Name).
		Resource("services").
		Name(rc.name).
		Suffix("ConsumeCPU").
		Param("milicores", strconv.Itoa(milicores)).
		Param("durationSec", strconv.Itoa(durationSec)).
		Do().
		Raw()
	expectNoError(err)
}

func (rc *ConsumingRC) CleanUp() {
	rc.stop <- 0
	expectNoError(DeleteRC(rc.framework.Client, rc.framework.Namespace.Name, rc.name))
	expectNoError(rc.framework.Client.Services(rc.framework.Namespace.Name).Delete(rc.name))
}

func startService(c *client.Client, ns, name string, replicas int) {
	c.Services(ns).Create(&api.Service{
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
