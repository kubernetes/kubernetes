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

package common

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

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
	resourceConsumerImage           = "gcr.io/google_containers/resource_consumer:beta4"
	resourceConsumerControllerImage = "gcr.io/google_containers/resource_consumer/controller:beta4"
	rcIsNil                         = "ERROR: replicationController = nil"
	deploymentIsNil                 = "ERROR: deployment = nil"
	rsIsNil                         = "ERROR: replicaset = nil"
	invalidKind                     = "ERROR: invalid workload kind for resource consumer"
	customMetricName                = "QPS"
	serviceInitializationTimeout    = 2 * time.Minute
	serviceInitializationInterval   = 15 * time.Second
)

const (
	KindRC         = "ReplicationController"
	KindDeployment = "Deployment"
	KindReplicaSet = "ReplicaSet"
	subresource    = "scale"
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
	controllerName           string
	kind                     string
	framework                *framework.Framework
	cpu                      chan int
	mem                      chan int
	customMetric             chan int
	stopCPU                  chan int
	stopMem                  chan int
	stopCustomMetric         chan int
	stopWaitGroup            sync.WaitGroup
	consumptionTimeInSeconds int
	sleepTime                time.Duration
	requestSizeInMillicores  int
	requestSizeInMegabytes   int
	requestSizeCustomMetric  int
}

func GetResourceConsumerImage() string {
	return resourceConsumerImage
}

func NewDynamicResourceConsumer(name, kind string, replicas, initCPUTotal, initMemoryTotal, initCustomMetric int, cpuLimit, memLimit int64, f *framework.Framework) *ResourceConsumer {
	return newResourceConsumer(name, kind, replicas, initCPUTotal, initMemoryTotal, initCustomMetric, dynamicConsumptionTimeInSeconds,
		dynamicRequestSizeInMillicores, dynamicRequestSizeInMegabytes, dynamicRequestSizeCustomMetric, cpuLimit, memLimit, f)
}

// TODO this still defaults to replication controller
func NewStaticResourceConsumer(name string, replicas, initCPUTotal, initMemoryTotal, initCustomMetric int, cpuLimit, memLimit int64, f *framework.Framework) *ResourceConsumer {
	return newResourceConsumer(name, KindRC, replicas, initCPUTotal, initMemoryTotal, initCustomMetric, staticConsumptionTimeInSeconds,
		initCPUTotal/replicas, initMemoryTotal/replicas, initCustomMetric/replicas, cpuLimit, memLimit, f)
}

/*
NewResourceConsumer creates new ResourceConsumer
initCPUTotal argument is in millicores
initMemoryTotal argument is in megabytes
memLimit argument is in megabytes, memLimit is a maximum amount of memory that can be consumed by a single pod
cpuLimit argument is in millicores, cpuLimit is a maximum amount of cpu that can be consumed by a single pod
*/
func newResourceConsumer(name, kind string, replicas, initCPUTotal, initMemoryTotal, initCustomMetric, consumptionTimeInSeconds, requestSizeInMillicores,
	requestSizeInMegabytes int, requestSizeCustomMetric int, cpuLimit, memLimit int64, f *framework.Framework) *ResourceConsumer {

	runServiceAndWorkloadForResourceConsumer(f.ClientSet, f.InternalClientset, f.Namespace.Name, name, kind, replicas, cpuLimit, memLimit)
	rc := &ResourceConsumer{
		name:                     name,
		controllerName:           name + "-ctrl",
		kind:                     kind,
		framework:                f,
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
	framework.Logf("RC %s: consume %v millicores in total", rc.name, millicores)
	rc.cpu <- millicores
}

// ConsumeMem consumes given number of Mem
func (rc *ResourceConsumer) ConsumeMem(megabytes int) {
	framework.Logf("RC %s: consume %v MB in total", rc.name, megabytes)
	rc.mem <- megabytes
}

// ConsumeMem consumes given number of custom metric
func (rc *ResourceConsumer) ConsumeCustomMetric(amount int) {
	framework.Logf("RC %s: consume custom metric %v in total", rc.name, amount)
	rc.customMetric <- amount
}

func (rc *ResourceConsumer) makeConsumeCPURequests() {
	defer GinkgoRecover()
	rc.stopWaitGroup.Add(1)
	defer rc.stopWaitGroup.Done()
	sleepTime := time.Duration(0)
	millicores := 0
	for {
		select {
		case millicores = <-rc.cpu:
			framework.Logf("RC %s: setting consumption to %v millicores in total", rc.name, millicores)
		case <-time.After(sleepTime):
			framework.Logf("RC %s: sending request to consume %d millicores", rc.name, millicores)
			rc.sendConsumeCPURequest(millicores)
			sleepTime = rc.sleepTime
		case <-rc.stopCPU:
			framework.Logf("RC %s: stopping CPU consumer", rc.name)
			return
		}
	}
}

func (rc *ResourceConsumer) makeConsumeMemRequests() {
	defer GinkgoRecover()
	rc.stopWaitGroup.Add(1)
	defer rc.stopWaitGroup.Done()
	sleepTime := time.Duration(0)
	megabytes := 0
	for {
		select {
		case megabytes = <-rc.mem:
			framework.Logf("RC %s: setting consumption to %v MB in total", rc.name, megabytes)
		case <-time.After(sleepTime):
			framework.Logf("RC %s: sending request to consume %d MB", rc.name, megabytes)
			rc.sendConsumeMemRequest(megabytes)
			sleepTime = rc.sleepTime
		case <-rc.stopMem:
			framework.Logf("RC %s: stopping mem consumer", rc.name)
			return
		}
	}
}

func (rc *ResourceConsumer) makeConsumeCustomMetric() {
	defer GinkgoRecover()
	rc.stopWaitGroup.Add(1)
	defer rc.stopWaitGroup.Done()
	sleepTime := time.Duration(0)
	delta := 0
	for {
		select {
		case delta := <-rc.customMetric:
			framework.Logf("RC %s: setting bump of metric %s to %d in total", rc.name, customMetricName, delta)
		case <-time.After(sleepTime):
			framework.Logf("RC %s: sending request to consume %d of custom metric %s", rc.name, delta, customMetricName)
			rc.sendConsumeCustomMetric(delta)
			sleepTime = rc.sleepTime
		case <-rc.stopCustomMetric:
			framework.Logf("RC %s: stopping metric consumer", rc.name)
			return
		}
	}
}

func (rc *ResourceConsumer) sendConsumeCPURequest(millicores int) {
	proxyRequest, err := framework.GetServicesProxyRequest(rc.framework.ClientSet, rc.framework.ClientSet.Core().RESTClient().Post())
	framework.ExpectNoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	err = wait.PollImmediate(serviceInitializationInterval, serviceInitializationTimeout, func() (bool, error) {
		req := proxyRequest.Namespace(rc.framework.Namespace.Name).
			Context(ctx).
			Name(rc.controllerName).
			Suffix("ConsumeCPU").
			Param("millicores", strconv.Itoa(millicores)).
			Param("durationSec", strconv.Itoa(rc.consumptionTimeInSeconds)).
			Param("requestSizeMillicores", strconv.Itoa(rc.requestSizeInMillicores))
		framework.Logf("ConsumeCPU URL: %v", *req.URL())
		_, err := req.DoRaw()
		if err != nil {
			framework.Logf("ConsumeCPU failure: %v", err)
			return false, nil
		}
		return true, nil
	})

	framework.ExpectNoError(err)
}

// sendConsumeMemRequest sends POST request for memory consumption
func (rc *ResourceConsumer) sendConsumeMemRequest(megabytes int) {
	proxyRequest, err := framework.GetServicesProxyRequest(rc.framework.ClientSet, rc.framework.ClientSet.Core().RESTClient().Post())
	framework.ExpectNoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	err = wait.PollImmediate(serviceInitializationInterval, serviceInitializationTimeout, func() (bool, error) {
		req := proxyRequest.Namespace(rc.framework.Namespace.Name).
			Context(ctx).
			Name(rc.controllerName).
			Suffix("ConsumeMem").
			Param("megabytes", strconv.Itoa(megabytes)).
			Param("durationSec", strconv.Itoa(rc.consumptionTimeInSeconds)).
			Param("requestSizeMegabytes", strconv.Itoa(rc.requestSizeInMegabytes))
		framework.Logf("ConsumeMem URL: %v", *req.URL())
		_, err := req.DoRaw()
		if err != nil {
			framework.Logf("ConsumeMem failure: %v", err)
			return false, nil
		}
		return true, nil
	})

	framework.ExpectNoError(err)
}

// sendConsumeCustomMetric sends POST request for custom metric consumption
func (rc *ResourceConsumer) sendConsumeCustomMetric(delta int) {
	proxyRequest, err := framework.GetServicesProxyRequest(rc.framework.ClientSet, rc.framework.ClientSet.Core().RESTClient().Post())
	framework.ExpectNoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	err = wait.PollImmediate(serviceInitializationInterval, serviceInitializationTimeout, func() (bool, error) {
		req := proxyRequest.Namespace(rc.framework.Namespace.Name).
			Context(ctx).
			Name(rc.controllerName).
			Suffix("BumpMetric").
			Param("metric", customMetricName).
			Param("delta", strconv.Itoa(delta)).
			Param("durationSec", strconv.Itoa(rc.consumptionTimeInSeconds)).
			Param("requestSizeMetrics", strconv.Itoa(rc.requestSizeCustomMetric))
		framework.Logf("ConsumeCustomMetric URL: %v", *req.URL())
		_, err := req.DoRaw()
		if err != nil {
			framework.Logf("ConsumeCustomMetric failure: %v", err)
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err)
}

func (rc *ResourceConsumer) GetReplicas() int {
	switch rc.kind {
	case KindRC:
		replicationController, err := rc.framework.ClientSet.Core().ReplicationControllers(rc.framework.Namespace.Name).Get(rc.name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		if replicationController == nil {
			framework.Failf(rcIsNil)
		}
		return int(replicationController.Status.ReadyReplicas)
	case KindDeployment:
		deployment, err := rc.framework.ClientSet.Extensions().Deployments(rc.framework.Namespace.Name).Get(rc.name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		if deployment == nil {
			framework.Failf(deploymentIsNil)
		}
		return int(deployment.Status.ReadyReplicas)
	case KindReplicaSet:
		rs, err := rc.framework.ClientSet.Extensions().ReplicaSets(rc.framework.Namespace.Name).Get(rc.name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		if rs == nil {
			framework.Failf(rsIsNil)
		}
		return int(rs.Status.ReadyReplicas)
	default:
		framework.Failf(invalidKind)
	}
	return 0
}

func (rc *ResourceConsumer) WaitForReplicas(desiredReplicas int, duration time.Duration) {
	interval := 20 * time.Second
	err := wait.PollImmediate(interval, duration, func() (bool, error) {
		replicas := rc.GetReplicas()
		framework.Logf("waiting for %d replicas (current: %d)", desiredReplicas, replicas)
		return replicas == desiredReplicas, nil // Expected number of replicas found. Exit.
	})
	framework.ExpectNoErrorWithOffset(1, err, "timeout waiting %v for %d replicas", duration, desiredReplicas)
}

func (rc *ResourceConsumer) EnsureDesiredReplicas(desiredReplicas int, duration time.Duration) {
	interval := 10 * time.Second
	err := wait.PollImmediate(interval, duration, func() (bool, error) {
		replicas := rc.GetReplicas()
		framework.Logf("expecting there to be %d replicas (are: %d)", desiredReplicas, replicas)
		if replicas != desiredReplicas {
			return false, fmt.Errorf("number of replicas changed unexpectedly")
		} else {
			return false, nil // Expected number of replicas found. Continue polling until timeout.
		}
	})
	// The call above always returns an error, but if it is timeout, it's OK (condition satisfied all the time).
	if err == wait.ErrWaitTimeout {
		framework.Logf("Number of replicas was stable over %v", duration)
		return
	}
	framework.ExpectNoErrorWithOffset(1, err)
}

// Pause stops background goroutines responsible for consuming resources.
func (rc *ResourceConsumer) Pause() {
	By(fmt.Sprintf("HPA pausing RC %s", rc.name))
	rc.stopCPU <- 0
	rc.stopMem <- 0
	rc.stopCustomMetric <- 0
	rc.stopWaitGroup.Wait()
}

// Pause starts background goroutines responsible for consuming resources.
func (rc *ResourceConsumer) Resume() {
	By(fmt.Sprintf("HPA resuming RC %s", rc.name))
	go rc.makeConsumeCPURequests()
	go rc.makeConsumeMemRequests()
	go rc.makeConsumeCustomMetric()
}

func (rc *ResourceConsumer) CleanUp() {
	By(fmt.Sprintf("Removing consuming RC %s", rc.name))
	close(rc.stopCPU)
	close(rc.stopMem)
	close(rc.stopCustomMetric)
	rc.stopWaitGroup.Wait()
	// Wait some time to ensure all child goroutines are finished.
	time.Sleep(10 * time.Second)
	kind, err := kindOf(rc.kind)
	framework.ExpectNoError(err)
	framework.ExpectNoError(framework.DeleteResourceAndPods(rc.framework.ClientSet, rc.framework.InternalClientset, kind, rc.framework.Namespace.Name, rc.name))
	framework.ExpectNoError(rc.framework.ClientSet.Core().Services(rc.framework.Namespace.Name).Delete(rc.name, nil))
	framework.ExpectNoError(framework.DeleteResourceAndPods(rc.framework.ClientSet, rc.framework.InternalClientset, api.Kind("ReplicationController"), rc.framework.Namespace.Name, rc.controllerName))
	framework.ExpectNoError(rc.framework.ClientSet.Core().Services(rc.framework.Namespace.Name).Delete(rc.controllerName, nil))
}

func kindOf(kind string) (schema.GroupKind, error) {
	switch kind {
	case KindRC:
		return api.Kind(kind), nil
	case KindReplicaSet:
		return extensionsinternal.Kind(kind), nil
	case KindDeployment:
		return extensionsinternal.Kind(kind), nil
	default:
		return schema.GroupKind{}, fmt.Errorf("Unsupported kind: %v", kind)
	}
}

func runServiceAndWorkloadForResourceConsumer(c clientset.Interface, internalClient internalclientset.Interface, ns, name, kind string, replicas int, cpuLimitMillis, memLimitMb int64) {
	By(fmt.Sprintf("Running consuming RC %s via %s with %v replicas", name, kind, replicas))
	_, err := c.Core().Services(ns).Create(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Port:       port,
				TargetPort: intstr.FromInt(targetPort),
			}},

			Selector: map[string]string{
				"name": name,
			},
		},
	})
	framework.ExpectNoError(err)

	rcConfig := testutils.RCConfig{
		Client:         c,
		InternalClient: internalClient,
		Image:          resourceConsumerImage,
		Name:           name,
		Namespace:      ns,
		Timeout:        timeoutRC,
		Replicas:       replicas,
		CpuRequest:     cpuLimitMillis,
		CpuLimit:       cpuLimitMillis,
		MemRequest:     memLimitMb * 1024 * 1024, // MemLimit is in bytes
		MemLimit:       memLimitMb * 1024 * 1024,
	}

	switch kind {
	case KindRC:
		framework.ExpectNoError(framework.RunRC(rcConfig))
		break
	case KindDeployment:
		dpConfig := testutils.DeploymentConfig{
			RCConfig: rcConfig,
		}
		framework.ExpectNoError(framework.RunDeployment(dpConfig))
		break
	case KindReplicaSet:
		rsConfig := testutils.ReplicaSetConfig{
			RCConfig: rcConfig,
		}
		By(fmt.Sprintf("creating replicaset %s in namespace %s", rsConfig.Name, rsConfig.Namespace))
		framework.ExpectNoError(framework.RunReplicaSet(rsConfig))
		break
	default:
		framework.Failf(invalidKind)
	}

	By(fmt.Sprintf("Running controller"))
	controllerName := name + "-ctrl"
	_, err = c.Core().Services(ns).Create(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: controllerName,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Port:       port,
				TargetPort: intstr.FromInt(targetPort),
			}},

			Selector: map[string]string{
				"name": controllerName,
			},
		},
	})
	framework.ExpectNoError(err)

	dnsClusterFirst := v1.DNSClusterFirst
	controllerRcConfig := testutils.RCConfig{
		Client:    c,
		Image:     resourceConsumerControllerImage,
		Name:      controllerName,
		Namespace: ns,
		Timeout:   timeoutRC,
		Replicas:  1,
		Command:   []string{"/controller", "--consumer-service-name=" + name, "--consumer-service-namespace=" + ns, "--consumer-port=80"},
		DNSPolicy: &dnsClusterFirst,
	}
	framework.ExpectNoError(framework.RunRC(controllerRcConfig))

	// Wait for endpoints to propagate for the controller service.
	framework.ExpectNoError(framework.WaitForServiceEndpointsNum(
		c, ns, controllerName, 1, startServiceInterval, startServiceTimeout))
}

func CreateCPUHorizontalPodAutoscaler(rc *ResourceConsumer, cpu, minReplicas, maxRepl int32) *autoscalingv1.HorizontalPodAutoscaler {
	hpa := &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      rc.name,
			Namespace: rc.framework.Namespace.Name,
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind: rc.kind,
				Name: rc.name,
			},
			MinReplicas:                    &minReplicas,
			MaxReplicas:                    maxRepl,
			TargetCPUUtilizationPercentage: &cpu,
		},
	}
	hpa, errHPA := rc.framework.ClientSet.Autoscaling().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	framework.ExpectNoError(errHPA)
	return hpa
}

func DeleteHorizontalPodAutoscaler(rc *ResourceConsumer, autoscalerName string) {
	rc.framework.ClientSet.Autoscaling().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Delete(autoscalerName, nil)
}
