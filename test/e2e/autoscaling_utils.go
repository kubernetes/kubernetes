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
	"fmt"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/intstr"

	. "github.com/onsi/ginkgo"
)

const (
	dynamicConsumptionTimeInSeconds = 30
	staticConsumptionTimeInSeconds  = 3600
	dynamicRequestSizeInMillicores  = 20
	dynamicRequestSizeInMegabytes   = 100
	dynamicRequestSizeCustomMetric  = 10
	port                            = 80
	targetPort                      = 8080
	timeoutRC                       = 120 * time.Second
	startServiceTimeout             = time.Minute
	startServiceInterval            = 5 * time.Second
	resourceConsumerImage           = "gcr.io/google_containers/resource_consumer:beta2"
	rcIsNil                         = "ERROR: replicationController = nil"
	deploymentIsNil                 = "ERROR: deployment = nil"
	rsIsNil                         = "ERROR: replicaset = nil"
	invalidKind                     = "ERROR: invalid workload kind for resource consumer"
	customMetricName                = "QPS"
)

/*
ResourceConsumer is a tool for testing. It helps create specified usage of CPU or memory (Warning: memory not supported)
typical use case:
rc.ConsumeCPU(600)
// ... check your assumption here
rc.ConsumeCPU(300)
// ... check your assumption here
*/
type ResourceConsumer struct {
	name                     string
	kind                     string
	framework                *Framework
	cpu                      chan int
	mem                      chan int
	customMetric             chan int
	stopCPU                  chan int
	stopMem                  chan int
	stopCustomMetric         chan int
	consumptionTimeInSeconds int
	sleepTime                time.Duration
	requestSizeInMillicores  int
	requestSizeInMegabytes   int
	requestSizeCustomMetric  int
}

func NewDynamicResourceConsumer(name, kind string, replicas, initCPUTotal, initMemoryTotal, initCustomMetric int, cpuLimit, memLimit int64, framework *Framework) *ResourceConsumer {
	return newResourceConsumer(name, kind, replicas, initCPUTotal, initMemoryTotal, initCustomMetric, dynamicConsumptionTimeInSeconds,
		dynamicRequestSizeInMillicores, dynamicRequestSizeInMegabytes, dynamicRequestSizeCustomMetric, cpuLimit, memLimit, framework)
}

// TODO this still defaults to replication controller
func NewStaticResourceConsumer(name string, replicas, initCPUTotal, initMemoryTotal, initCustomMetric int, cpuLimit, memLimit int64, framework *Framework) *ResourceConsumer {
	return newResourceConsumer(name, kindRC, replicas, initCPUTotal, initMemoryTotal, initCustomMetric, staticConsumptionTimeInSeconds,
		initCPUTotal/replicas, initMemoryTotal/replicas, initCustomMetric/replicas, cpuLimit, memLimit, framework)
}

/*
NewResourceConsumer creates new ResourceConsumer
initCPUTotal argument is in millicores
initMemoryTotal argument is in megabytes
memLimit argument is in megabytes, memLimit is a maximum amount of memory that can be consumed by a single pod
cpuLimit argument is in millicores, cpuLimit is a maximum amount of cpu that can be consumed by a single pod
*/
func newResourceConsumer(name, kind string, replicas, initCPUTotal, initMemoryTotal, initCustomMetric, consumptionTimeInSeconds, requestSizeInMillicores,
	requestSizeInMegabytes int, requestSizeCustomMetric int, cpuLimit, memLimit int64, framework *Framework) *ResourceConsumer {

	runServiceAndWorkloadForResourceConsumer(framework.Client, framework.Namespace.Name, name, kind, replicas, cpuLimit, memLimit)
	rc := &ResourceConsumer{
		name:                     name,
		kind:                     kind,
		framework:                framework,
		cpu:                      make(chan int),
		mem:                      make(chan int),
		customMetric:             make(chan int),
		stopCPU:                  make(chan int),
		stopMem:                  make(chan int),
		stopCustomMetric:         make(chan int),
		consumptionTimeInSeconds: consumptionTimeInSeconds,
		sleepTime:                time.Duration(consumptionTimeInSeconds) * time.Second,
		requestSizeInMillicores:  requestSizeInMillicores,
		requestSizeInMegabytes:   requestSizeInMegabytes,
		requestSizeCustomMetric:  requestSizeCustomMetric,
	}
	go rc.makeConsumeCPURequests()
	rc.ConsumeCPU(initCPUTotal)
	go rc.makeConsumeMemRequests()
	rc.ConsumeMem(initMemoryTotal)
	go rc.makeConsumeCustomMetric()
	rc.ConsumeCustomMetric(initCustomMetric)
	return rc
}

// ConsumeCPU consumes given number of CPU
func (rc *ResourceConsumer) ConsumeCPU(millicores int) {
	Logf("RC %s: consume %v millicores in total", rc.name, millicores)
	rc.cpu <- millicores
}

// ConsumeMem consumes given number of Mem
func (rc *ResourceConsumer) ConsumeMem(megabytes int) {
	Logf("RC %s: consume %v MB in total", rc.name, megabytes)
	rc.mem <- megabytes
}

// ConsumeMem consumes given number of custom metric
func (rc *ResourceConsumer) ConsumeCustomMetric(amount int) {
	Logf("RC %s: consume custom metric %v in total", rc.name, amount)
	rc.customMetric <- amount
}

func (rc *ResourceConsumer) makeConsumeCPURequests() {
	defer GinkgoRecover()
	var count int
	var rest int
	sleepTime := time.Duration(0)
	for {
		select {
		case millicores := <-rc.cpu:
			Logf("RC %s: consume %v millicores in total", rc.name, millicores)
			if rc.requestSizeInMillicores != 0 {
				count = millicores / rc.requestSizeInMillicores
			}
			rest = millicores - count*rc.requestSizeInMillicores
		case <-time.After(sleepTime):
			Logf("RC %s: sending %v requests to consume %v millicores each and 1 request to consume %v millicores", rc.name, count, rc.requestSizeInMillicores, rest)
			if count > 0 {
				rc.sendConsumeCPURequests(count, rc.requestSizeInMillicores, rc.consumptionTimeInSeconds)
			}
			if rest > 0 {
				go rc.sendOneConsumeCPURequest(rest, rc.consumptionTimeInSeconds)
			}
			sleepTime = rc.sleepTime
		case <-rc.stopCPU:
			return
		}
	}
}

func (rc *ResourceConsumer) makeConsumeMemRequests() {
	defer GinkgoRecover()
	var count int
	var rest int
	sleepTime := time.Duration(0)
	for {
		select {
		case megabytes := <-rc.mem:
			Logf("RC %s: consume %v MB in total", rc.name, megabytes)
			if rc.requestSizeInMegabytes != 0 {
				count = megabytes / rc.requestSizeInMegabytes
			}
			rest = megabytes - count*rc.requestSizeInMegabytes
		case <-time.After(sleepTime):
			Logf("RC %s: sending %v requests to consume %v MB each and 1 request to consume %v MB", rc.name, count, rc.requestSizeInMegabytes, rest)
			if count > 0 {
				rc.sendConsumeMemRequests(count, rc.requestSizeInMegabytes, rc.consumptionTimeInSeconds)
			}
			if rest > 0 {
				go rc.sendOneConsumeMemRequest(rest, rc.consumptionTimeInSeconds)
			}
			sleepTime = rc.sleepTime
		case <-rc.stopMem:
			return
		}
	}
}

func (rc *ResourceConsumer) makeConsumeCustomMetric() {
	defer GinkgoRecover()
	var count int
	var rest int
	sleepTime := time.Duration(0)
	for {
		select {
		case total := <-rc.customMetric:
			Logf("RC %s: consume custom metric %v in total", rc.name, total)
			if rc.requestSizeInMegabytes != 0 {
				count = total / rc.requestSizeCustomMetric
			}
			rest = total - count*rc.requestSizeCustomMetric
		case <-time.After(sleepTime):
			Logf("RC %s: sending %v requests to consume %v custom metric each and 1 request to consume %v",
				rc.name, count, rc.requestSizeCustomMetric, rest)
			if count > 0 {
				rc.sendConsumeCustomMetric(count, rc.requestSizeCustomMetric, rc.consumptionTimeInSeconds)
			}
			if rest > 0 {
				go rc.sendOneConsumeCustomMetric(rest, rc.consumptionTimeInSeconds)
			}
			sleepTime = rc.sleepTime
		case <-rc.stopCustomMetric:
			return
		}
	}
}

func (rc *ResourceConsumer) sendConsumeCPURequests(requests, millicores, durationSec int) {
	for i := 0; i < requests; i++ {
		go rc.sendOneConsumeCPURequest(millicores, durationSec)
	}
}

func (rc *ResourceConsumer) sendConsumeMemRequests(requests, megabytes, durationSec int) {
	for i := 0; i < requests; i++ {
		go rc.sendOneConsumeMemRequest(megabytes, durationSec)
	}
}

func (rc *ResourceConsumer) sendConsumeCustomMetric(requests, delta, durationSec int) {
	for i := 0; i < requests; i++ {
		go rc.sendOneConsumeCustomMetric(delta, durationSec)
	}
}

// sendOneConsumeCPURequest sends POST request for cpu consumption
func (rc *ResourceConsumer) sendOneConsumeCPURequest(millicores int, durationSec int) {
	defer GinkgoRecover()
	proxyRequest, err := getServicesProxyRequest(rc.framework.Client, rc.framework.Client.Post())
	expectNoError(err)
	_, err = proxyRequest.Namespace(rc.framework.Namespace.Name).
		Name(rc.name).
		Suffix("ConsumeCPU").
		Param("millicores", strconv.Itoa(millicores)).
		Param("durationSec", strconv.Itoa(durationSec)).
		DoRaw()
	expectNoError(err)
}

// sendOneConsumeMemRequest sends POST request for memory consumption
func (rc *ResourceConsumer) sendOneConsumeMemRequest(megabytes int, durationSec int) {
	defer GinkgoRecover()
	proxyRequest, err := getServicesProxyRequest(rc.framework.Client, rc.framework.Client.Post())
	expectNoError(err)
	_, err = proxyRequest.Namespace(rc.framework.Namespace.Name).
		Name(rc.name).
		Suffix("ConsumeMem").
		Param("megabytes", strconv.Itoa(megabytes)).
		Param("durationSec", strconv.Itoa(durationSec)).
		DoRaw()
	expectNoError(err)
}

// sendOneConsumeCustomMetric sends POST request for custom metric consumption
func (rc *ResourceConsumer) sendOneConsumeCustomMetric(delta int, durationSec int) {
	defer GinkgoRecover()
	proxyRequest, err := getServicesProxyRequest(rc.framework.Client, rc.framework.Client.Post())
	expectNoError(err)
	_, err = proxyRequest.Namespace(rc.framework.Namespace.Name).
		Name(rc.name).
		Suffix("BumpMetric").
		Param("metric", customMetricName).
		Param("delta", strconv.Itoa(delta)).
		Param("durationSec", strconv.Itoa(durationSec)).
		DoRaw()
	expectNoError(err)
}

func (rc *ResourceConsumer) GetReplicas() int {
	switch rc.kind {
	case kindRC:
		replicationController, err := rc.framework.Client.ReplicationControllers(rc.framework.Namespace.Name).Get(rc.name)
		expectNoError(err)
		if replicationController == nil {
			Failf(rcIsNil)
		}
		return replicationController.Status.Replicas
	case kindDeployment:
		deployment, err := rc.framework.Client.Deployments(rc.framework.Namespace.Name).Get(rc.name)
		expectNoError(err)
		if deployment == nil {
			Failf(deploymentIsNil)
		}
		return deployment.Status.Replicas
	case kindReplicaSet:
		rs, err := rc.framework.Client.ReplicaSets(rc.framework.Namespace.Name).Get(rc.name)
		expectNoError(err)
		if rs == nil {
			Failf(rsIsNil)
		}
		return rs.Status.Replicas
	default:
		Failf(invalidKind)
	}
	return 0
}

func (rc *ResourceConsumer) WaitForReplicas(desiredReplicas int) {
	timeout := 10 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		if desiredReplicas == rc.GetReplicas() {
			Logf("%s: current replicas number is equal to desired replicas number: %d", rc.kind, desiredReplicas)
			return
		} else {
			Logf("%s: current replicas number %d waiting to be %d", rc.kind, rc.GetReplicas(), desiredReplicas)
		}
	}
	Failf("timeout waiting %v for pods size to be %d", timeout, desiredReplicas)
}

func (rc *ResourceConsumer) EnsureDesiredReplicas(desiredReplicas int, timeout time.Duration) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(10 * time.Second) {
		actual := rc.GetReplicas()
		if desiredReplicas != actual {
			Failf("Number of replicas has changed: expected %v, got %v", desiredReplicas, actual)
		}
		Logf("Number of replicas is as expected")
	}
	Logf("Number of replicas was stable over %v", timeout)
}

func (rc *ResourceConsumer) CleanUp() {
	By(fmt.Sprintf("Removing consuming RC %s", rc.name))
	rc.stopCPU <- 0
	rc.stopMem <- 0
	rc.stopCustomMetric <- 0
	// Wait some time to ensure all child goroutines are finished.
	time.Sleep(10 * time.Second)
	expectNoError(DeleteRC(rc.framework.Client, rc.framework.Namespace.Name, rc.name))
	expectNoError(rc.framework.Client.Services(rc.framework.Namespace.Name).Delete(rc.name))
}

func runServiceAndWorkloadForResourceConsumer(c *client.Client, ns, name, kind string, replicas int, cpuLimitMillis, memLimitMb int64) {
	By(fmt.Sprintf("Running consuming RC %s via %s with %v replicas", name, kind, replicas))
	_, err := c.Services(ns).Create(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{
				Port:       port,
				TargetPort: intstr.FromInt(targetPort),
			}},

			Selector: map[string]string{
				"name": name,
			},
		},
	})
	expectNoError(err)

	rcConfig := RCConfig{
		Client:     c,
		Image:      resourceConsumerImage,
		Name:       name,
		Namespace:  ns,
		Timeout:    timeoutRC,
		Replicas:   replicas,
		CpuRequest: cpuLimitMillis,
		CpuLimit:   cpuLimitMillis,
		MemRequest: memLimitMb * 1024 * 1024, // MemLimit is in bytes
		MemLimit:   memLimitMb * 1024 * 1024,
	}

	switch kind {
	case kindRC:
		expectNoError(RunRC(rcConfig))
		break
	case kindDeployment:
		dpConfig := DeploymentConfig{
			rcConfig,
		}
		expectNoError(RunDeployment(dpConfig))
		break
	case kindReplicaSet:
		rsConfig := ReplicaSetConfig{
			rcConfig,
		}
		expectNoError(RunReplicaSet(rsConfig))
		break
	default:
		Failf(invalidKind)
	}

	// Make sure endpoints are propagated.
	// TODO(piosz): replace sleep with endpoints watch.
	time.Sleep(10 * time.Second)
}
