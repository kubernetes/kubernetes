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

package autoscaling

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	crdclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edebug "k8s.io/kubernetes/test/e2e/framework/debug"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/utils/ptr"

	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	dynamicConsumptionTimeInSeconds = 30
	dynamicRequestSizeInMillicores  = 100
	dynamicRequestSizeInMegabytes   = 100
	dynamicRequestSizeCustomMetric  = 10
	port                            = 80
	portName                        = "http"
	targetPort                      = 8080
	sidecarTargetPort               = 8081
	timeoutRC                       = 120 * time.Second
	invalidKind                     = "ERROR: invalid workload kind for resource consumer"
	customMetricName                = "QPS"
	serviceInitializationTimeout    = 2 * time.Minute
	serviceInitializationInterval   = 15 * time.Second
	megabytes                       = 1024 * 1024
	crdVersion                      = "v1"
	crdKind                         = "TestCRD"
	crdGroup                        = "autoscalinge2e.example.com"
	crdName                         = "testcrd"
	crdNamePlural                   = "testcrds"
)

var (
	// KindRC is the GVK for ReplicationController
	KindRC = schema.GroupVersionKind{Version: "v1", Kind: "ReplicationController"}
	// KindDeployment is the GVK for Deployment
	KindDeployment = schema.GroupVersionKind{Group: "apps", Version: "v1beta2", Kind: "Deployment"}
	// KindReplicaSet is the GVK for ReplicaSet
	KindReplicaSet = schema.GroupVersionKind{Group: "apps", Version: "v1beta2", Kind: "ReplicaSet"}
	// KindCRD is the GVK for CRD for test purposes
	KindCRD = schema.GroupVersionKind{Group: crdGroup, Version: crdVersion, Kind: crdKind}
)

// ScalingDirection identifies the scale direction for HPA Behavior.
type ScalingDirection int

const (
	DirectionUnknown ScalingDirection = iota
	ScaleUpDirection
	ScaleDownDirection
)

/*
ResourceConsumer is a tool for testing. It helps to create a specified usage of CPU or memory.
Typical use case:
rc.ConsumeCPU(600)
// ... check your assumption here
rc.ConsumeCPU(300)
// ... check your assumption here
*/
type ResourceConsumer struct {
	name                     string
	controllerName           string
	kind                     schema.GroupVersionKind
	nsName                   string
	clientSet                clientset.Interface
	apiExtensionClient       crdclientset.Interface
	dynamicClient            dynamic.Interface
	resourceClient           dynamic.ResourceInterface
	scaleClient              scaleclient.ScalesGetter
	customMetricName         string
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
	sidecarStatus            SidecarStatusType
	sidecarType              SidecarWorkloadType
}

// NewDynamicResourceConsumer is a wrapper to create a new dynamic ResourceConsumer
func NewDynamicResourceConsumer(ctx context.Context, name, nsName string, kind schema.GroupVersionKind, replicas, initCPUTotal, initMemoryTotal, initCustomMetric int, cpuLimit, memLimit int64, clientset clientset.Interface, scaleClient scaleclient.ScalesGetter, enableSidecar SidecarStatusType, sidecarType SidecarWorkloadType, podResources *v1.ResourceRequirements) *ResourceConsumer {
	return NewResourceConsumer(ctx, name, nsName, kind, replicas, customMetricName, initCPUTotal, initMemoryTotal, initCustomMetric, dynamicConsumptionTimeInSeconds,
		dynamicRequestSizeInMillicores, dynamicRequestSizeInMegabytes, dynamicRequestSizeCustomMetric, cpuLimit, memLimit, clientset, scaleClient, nil, nil, enableSidecar, sidecarType, podResources)
}

// getSidecarContainer returns sidecar container
func getSidecarContainer(name string, cpuLimit, memLimit int64) v1.Container {
	container := v1.Container{
		Name:    name + "-sidecar",
		Image:   imageutils.GetE2EImage(imageutils.ResourceConsumer),
		Command: []string{"/consumer", "-port=8081"},
		Ports:   []v1.ContainerPort{{ContainerPort: 80}},
	}

	if cpuLimit > 0 || memLimit > 0 {
		container.Resources.Limits = v1.ResourceList{}
		container.Resources.Requests = v1.ResourceList{}
	}

	if cpuLimit > 0 {
		container.Resources.Limits[v1.ResourceCPU] = *resource.NewMilliQuantity(cpuLimit, resource.DecimalSI)
		container.Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(cpuLimit, resource.DecimalSI)
	}

	if memLimit > 0 {
		container.Resources.Limits[v1.ResourceMemory] = *resource.NewQuantity(memLimit*megabytes, resource.DecimalSI)
		container.Resources.Requests[v1.ResourceMemory] = *resource.NewQuantity(memLimit*megabytes, resource.DecimalSI)
	}

	return container
}

/*
NewResourceConsumer creates new ResourceConsumer
initCPUTotal argument is in millicores
initMemoryTotal argument is in megabytes
memLimit argument is in megabytes, memLimit is a maximum amount of memory that can be consumed by a single pod
cpuLimit argument is in millicores, cpuLimit is a maximum amount of cpu that can be consumed by a single pod
*/
func NewResourceConsumer(ctx context.Context, name, nsName string, kind schema.GroupVersionKind, replicas int, customMetricName string, initCPUTotal, initMemoryTotal, initCustomMetric, consumptionTimeInSeconds, requestSizeInMillicores,
	requestSizeInMegabytes int, requestSizeCustomMetric int, cpuLimit, memLimit int64, clientset clientset.Interface, scaleClient scaleclient.ScalesGetter, podAnnotations, serviceAnnotations map[string]string, sidecarStatus SidecarStatusType, sidecarType SidecarWorkloadType, podResources *v1.ResourceRequirements) *ResourceConsumer {
	if podAnnotations == nil {
		podAnnotations = make(map[string]string)
	}
	if serviceAnnotations == nil {
		serviceAnnotations = make(map[string]string)
	}

	var additionalContainers []v1.Container

	if sidecarStatus == Enable {
		sidecarContainer := getSidecarContainer(name, cpuLimit, memLimit)
		additionalContainers = append(additionalContainers, sidecarContainer)
	}

	config, err := framework.LoadConfig()
	framework.ExpectNoError(err)
	apiExtensionClient, err := crdclientset.NewForConfig(config)
	framework.ExpectNoError(err)
	dynamicClient, err := dynamic.NewForConfig(config)
	framework.ExpectNoError(err)
	resourceClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crdGroup, Version: crdVersion, Resource: crdNamePlural}).Namespace(nsName)

	runServiceAndWorkloadForResourceConsumer(ctx, clientset, resourceClient, apiExtensionClient, nsName, name, kind, replicas, cpuLimit, memLimit, podAnnotations, serviceAnnotations, additionalContainers, podResources)
	controllerName := name + "-ctrl"
	// If sidecar is enabled and busy, run service and consumer for sidecar
	if sidecarStatus == Enable && sidecarType == Busy {
		runServiceAndSidecarForResourceConsumer(ctx, clientset, nsName, name, kind, replicas, serviceAnnotations)
		controllerName = name + "-sidecar-ctrl"
	}

	rc := &ResourceConsumer{
		name:                     name,
		controllerName:           controllerName,
		kind:                     kind,
		nsName:                   nsName,
		clientSet:                clientset,
		apiExtensionClient:       apiExtensionClient,
		scaleClient:              scaleClient,
		resourceClient:           resourceClient,
		dynamicClient:            dynamicClient,
		customMetricName:         customMetricName,
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
		sidecarType:              sidecarType,
		sidecarStatus:            sidecarStatus,
	}

	go rc.makeConsumeCPURequests(ctx)
	rc.ConsumeCPU(initCPUTotal)
	go rc.makeConsumeMemRequests(ctx)
	rc.ConsumeMem(initMemoryTotal)
	go rc.makeConsumeCustomMetric(ctx)
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

// ConsumeCustomMetric consumes given number of custom metric
func (rc *ResourceConsumer) ConsumeCustomMetric(amount int) {
	framework.Logf("RC %s: consume custom metric %v in total", rc.name, amount)
	rc.customMetric <- amount
}

func (rc *ResourceConsumer) makeConsumeCPURequests(ctx context.Context) {
	defer ginkgo.GinkgoRecover()
	rc.stopWaitGroup.Add(1)
	defer rc.stopWaitGroup.Done()
	tick := time.After(time.Duration(0))
	millicores := 0
	for {
		select {
		case millicores = <-rc.cpu:
			if millicores != 0 {
				framework.Logf("RC %s: setting consumption to %v millicores in total", rc.name, millicores)
			} else {
				framework.Logf("RC %s: disabling CPU consumption", rc.name)
			}
		case <-tick:
			if millicores != 0 {
				framework.Logf("RC %s: sending request to consume %d millicores", rc.name, millicores)
				rc.sendConsumeCPURequest(ctx, millicores)
			}
			tick = time.After(rc.sleepTime)
		case <-ctx.Done():
			framework.Logf("RC %s: stopping CPU consumer: %v", rc.name, ctx.Err())
			return
		case <-rc.stopCPU:
			framework.Logf("RC %s: stopping CPU consumer", rc.name)
			return
		}
	}
}

func (rc *ResourceConsumer) makeConsumeMemRequests(ctx context.Context) {
	defer ginkgo.GinkgoRecover()
	rc.stopWaitGroup.Add(1)
	defer rc.stopWaitGroup.Done()
	tick := time.After(time.Duration(0))
	megabytes := 0
	for {
		select {
		case megabytes = <-rc.mem:
			if megabytes != 0 {
				framework.Logf("RC %s: setting consumption to %v MB in total", rc.name, megabytes)
			} else {
				framework.Logf("RC %s: disabling mem consumption", rc.name)
			}
		case <-tick:
			if megabytes != 0 {
				framework.Logf("RC %s: sending request to consume %d MB", rc.name, megabytes)
				rc.sendConsumeMemRequest(ctx, megabytes)
			}
			tick = time.After(rc.sleepTime)
		case <-ctx.Done():
			framework.Logf("RC %s: stopping mem consumer: %v", rc.name, ctx.Err())
			return
		case <-rc.stopMem:
			framework.Logf("RC %s: stopping mem consumer", rc.name)
			return
		}
	}
}

func (rc *ResourceConsumer) makeConsumeCustomMetric(ctx context.Context) {
	defer ginkgo.GinkgoRecover()
	rc.stopWaitGroup.Add(1)
	defer rc.stopWaitGroup.Done()
	tick := time.After(time.Duration(0))
	delta := 0
	for {
		select {
		case delta = <-rc.customMetric:
			if delta != 0 {
				framework.Logf("RC %s: setting bump of metric %s to %d in total", rc.name, rc.customMetricName, delta)
			} else {
				framework.Logf("RC %s: disabling consumption of custom metric %s", rc.name, rc.customMetricName)
			}
		case <-tick:
			if delta != 0 {
				framework.Logf("RC %s: sending request to consume %d of custom metric %s", rc.name, delta, rc.customMetricName)
				rc.sendConsumeCustomMetric(ctx, delta)
			}
			tick = time.After(rc.sleepTime)
		case <-ctx.Done():
			framework.Logf("RC %s: stopping metric consumer: %v", rc.name, ctx.Err())
			return
		case <-rc.stopCustomMetric:
			framework.Logf("RC %s: stopping metric consumer", rc.name)
			return
		}
	}
}

func (rc *ResourceConsumer) sendConsumeCPURequest(ctx context.Context, millicores int) {
	err := framework.Gomega().Eventually(ctx, func(ctx context.Context) error {
		proxyRequest, err := e2eservice.GetServicesProxyRequest(rc.clientSet, rc.clientSet.CoreV1().RESTClient().Post())
		if err != nil {
			return err
		}
		req := proxyRequest.Namespace(rc.nsName).
			Name(fmt.Sprintf("%s:%s", rc.controllerName, portName)).
			Suffix("ConsumeCPU").
			Param("millicores", strconv.Itoa(millicores)).
			Param("durationSec", strconv.Itoa(rc.consumptionTimeInSeconds)).
			Param("requestSizeMillicores", strconv.Itoa(rc.requestSizeInMillicores))
		framework.Logf("ConsumeCPU URL: %v", *req.URL())
		_, err = req.DoRaw(ctx)
		if err != nil {
			framework.Logf("ConsumeCPU failure: %v", err)
			return err
		}
		return nil
	}).WithTimeout(serviceInitializationTimeout).WithPolling(serviceInitializationInterval).Should(gomega.Succeed())

	// Test has already finished (ctx got canceled), so don't fail on err from PollUntilContextTimeout
	// which is a side-effect to context cancelling from the cleanup task.
	if ctx.Err() != nil {
		return
	}

	framework.ExpectNoError(err)
}

// sendConsumeMemRequest sends POST request for memory consumption
func (rc *ResourceConsumer) sendConsumeMemRequest(ctx context.Context, megabytes int) {
	err := framework.Gomega().Eventually(ctx, func(ctx context.Context) error {
		proxyRequest, err := e2eservice.GetServicesProxyRequest(rc.clientSet, rc.clientSet.CoreV1().RESTClient().Post())
		if err != nil {
			return err
		}
		req := proxyRequest.Namespace(rc.nsName).
			Name(fmt.Sprintf("%s:%s", rc.controllerName, portName)).
			Suffix("ConsumeMem").
			Param("megabytes", strconv.Itoa(megabytes)).
			Param("durationSec", strconv.Itoa(rc.consumptionTimeInSeconds)).
			Param("requestSizeMegabytes", strconv.Itoa(rc.requestSizeInMegabytes))
		framework.Logf("ConsumeMem URL: %v", *req.URL())
		_, err = req.DoRaw(ctx)
		if err != nil {
			framework.Logf("ConsumeMem failure: %v", err)
			return err
		}
		return nil
	}).WithTimeout(serviceInitializationTimeout).WithPolling(serviceInitializationInterval).Should(gomega.Succeed())

	// Test has already finished (ctx got canceled), so don't fail on err from PollUntilContextTimeout
	// which is a side-effect to context cancelling from the cleanup task.
	if ctx.Err() != nil {
		return
	}

	framework.ExpectNoError(err)
}

// sendConsumeCustomMetric sends POST request for custom metric consumption
func (rc *ResourceConsumer) sendConsumeCustomMetric(ctx context.Context, delta int) {
	err := framework.Gomega().Eventually(ctx, func(ctx context.Context) error {
		proxyRequest, err := e2eservice.GetServicesProxyRequest(rc.clientSet, rc.clientSet.CoreV1().RESTClient().Post())
		if err != nil {
			return err
		}
		req := proxyRequest.Namespace(rc.nsName).
			Name(fmt.Sprintf("%s:%s", rc.controllerName, portName)).
			Suffix("BumpMetric").
			Param("metric", rc.customMetricName).
			Param("delta", strconv.Itoa(delta)).
			Param("durationSec", strconv.Itoa(rc.consumptionTimeInSeconds)).
			Param("requestSizeMetrics", strconv.Itoa(rc.requestSizeCustomMetric))
		framework.Logf("ConsumeCustomMetric URL: %v", *req.URL())
		_, err = req.DoRaw(ctx)
		if err != nil {
			framework.Logf("ConsumeCustomMetric failure: %v", err)
			return err
		}
		return nil
	}).WithTimeout(serviceInitializationTimeout).WithPolling(serviceInitializationInterval).Should(gomega.Succeed())

	// Test has already finished (ctx got canceled), so don't fail on err from PollUntilContextTimeout
	// which is a side-effect to context cancelling from the cleanup task.
	if ctx.Err() != nil {
		return
	}

	framework.ExpectNoError(err)
}

// GetReplicas get the replicas
func (rc *ResourceConsumer) GetReplicas(ctx context.Context) (int, error) {
	switch rc.kind {
	case KindRC:
		replicationController, err := rc.clientSet.CoreV1().ReplicationControllers(rc.nsName).Get(ctx, rc.name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		return int(replicationController.Status.ReadyReplicas), nil
	case KindDeployment:
		deployment, err := rc.clientSet.AppsV1().Deployments(rc.nsName).Get(ctx, rc.name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		return int(deployment.Status.ReadyReplicas), nil
	case KindReplicaSet:
		rs, err := rc.clientSet.AppsV1().ReplicaSets(rc.nsName).Get(ctx, rc.name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		return int(rs.Status.ReadyReplicas), nil
	case KindCRD:
		deployment, err := rc.clientSet.AppsV1().Deployments(rc.nsName).Get(ctx, rc.name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		deploymentReplicas := int64(deployment.Status.ReadyReplicas)

		scale, err := rc.scaleClient.Scales(rc.nsName).Get(ctx, schema.GroupResource{Group: crdGroup, Resource: crdNamePlural}, rc.name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		crdInstance, err := rc.resourceClient.Get(ctx, rc.name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		// Update custom resource's status.replicas with child Deployment's current number of ready replicas.
		err = unstructured.SetNestedField(crdInstance.Object, deploymentReplicas, "status", "replicas")
		if err != nil {
			return 0, err
		}
		_, err = rc.resourceClient.Update(ctx, crdInstance, metav1.UpdateOptions{})
		if err != nil {
			return 0, err
		}
		return int(scale.Spec.Replicas), nil
	default:
		return 0, fmt.Errorf(invalidKind)
	}
}

// GetHpa get the corresponding horizontalPodAutoscaler object
func (rc *ResourceConsumer) GetHpa(ctx context.Context, name string) (*autoscalingv1.HorizontalPodAutoscaler, error) {
	return rc.clientSet.AutoscalingV1().HorizontalPodAutoscalers(rc.nsName).Get(ctx, name, metav1.GetOptions{})
}

// WaitForReplicas wait for the desired replicas
func (rc *ResourceConsumer) WaitForReplicas(ctx context.Context, desiredReplicas int, duration time.Duration) {
	interval := 20 * time.Second
	err := framework.Gomega().Eventually(ctx, framework.HandleRetry(rc.GetReplicas)).
		WithTimeout(duration).
		WithPolling(interval).
		Should(gomega.Equal(desiredReplicas))

	framework.ExpectNoErrorWithOffset(1, err, "timeout waiting %v for %d replicas", duration, desiredReplicas)
}

// EnsureDesiredReplicasInRange ensure the replicas is in a desired range
func (rc *ResourceConsumer) EnsureDesiredReplicasInRange(ctx context.Context, minDesiredReplicas, maxDesiredReplicas int, duration time.Duration, hpaName string) {
	interval := 10 * time.Second
	desiredReplicasErr := framework.Gomega().Consistently(ctx, framework.HandleRetry(rc.GetReplicas)).
		WithTimeout(duration).
		WithPolling(interval).
		Should(gomega.And(gomega.BeNumerically(">=", minDesiredReplicas), gomega.BeNumerically("<=", maxDesiredReplicas)))

	// dump HPA for debugging
	as, err := rc.GetHpa(ctx, hpaName)
	if err != nil {
		framework.Logf("Error getting HPA: %s", err)
	} else {
		framework.Logf("HPA status: %+v", as.Status)
	}
	framework.ExpectNoError(desiredReplicasErr)
}

// Pause stops background goroutines responsible for consuming resources.
func (rc *ResourceConsumer) Pause() {
	ginkgo.By(fmt.Sprintf("HPA pausing RC %s", rc.name))
	rc.stopCPU <- 0
	rc.stopMem <- 0
	rc.stopCustomMetric <- 0
	rc.stopWaitGroup.Wait()
}

// Resume starts background goroutines responsible for consuming resources.
func (rc *ResourceConsumer) Resume(ctx context.Context) {
	ginkgo.By(fmt.Sprintf("HPA resuming RC %s", rc.name))
	go rc.makeConsumeCPURequests(ctx)
	go rc.makeConsumeMemRequests(ctx)
	go rc.makeConsumeCustomMetric(ctx)
}

// CleanUp clean up the background goroutines responsible for consuming resources.
func (rc *ResourceConsumer) CleanUp(ctx context.Context) {
	ginkgo.By(fmt.Sprintf("Removing consuming RC %s", rc.name))
	close(rc.stopCPU)
	close(rc.stopMem)
	close(rc.stopCustomMetric)
	rc.stopWaitGroup.Wait()
	// Wait some time to ensure all child goroutines are finished.
	time.Sleep(10 * time.Second)
	kind := rc.kind.GroupKind()
	if kind.Kind == crdKind {
		gvr := schema.GroupVersionResource{Group: crdGroup, Version: crdVersion, Resource: crdNamePlural}
		framework.ExpectNoError(e2eresource.DeleteCustomResourceAndWaitForGC(ctx, rc.clientSet, rc.dynamicClient, rc.scaleClient, gvr, rc.nsName, rc.name))

	} else {
		framework.ExpectNoError(e2eresource.DeleteResourceAndWaitForGC(ctx, rc.clientSet, kind, rc.nsName, rc.name))
	}

	framework.ExpectNoError(rc.clientSet.CoreV1().Services(rc.nsName).Delete(ctx, rc.name, metav1.DeleteOptions{}))
	framework.ExpectNoError(e2eresource.DeleteResourceAndWaitForGC(ctx, rc.clientSet, schema.GroupKind{Kind: "ReplicationController"}, rc.nsName, rc.controllerName))
	framework.ExpectNoError(rc.clientSet.CoreV1().Services(rc.nsName).Delete(ctx, rc.name+"-ctrl", metav1.DeleteOptions{}))
	// Cleanup sidecar related resources
	if rc.sidecarStatus == Enable && rc.sidecarType == Busy {
		framework.ExpectNoError(rc.clientSet.CoreV1().Services(rc.nsName).Delete(ctx, rc.name+"-sidecar", metav1.DeleteOptions{}))
		framework.ExpectNoError(rc.clientSet.CoreV1().Services(rc.nsName).Delete(ctx, rc.name+"-sidecar-ctrl", metav1.DeleteOptions{}))
	}
}

func createService(ctx context.Context, c clientset.Interface, name, ns string, annotations, selectors map[string]string, port int32, targetPort int) (*v1.Service, error) {
	return c.CoreV1().Services(ns).Create(ctx, &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: annotations,
			Labels:      map[string]string{"name": name},
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Name:       portName,
				Port:       port,
				TargetPort: intstr.FromInt32(int32(targetPort)),
			}},
			Selector: selectors,
		},
	}, metav1.CreateOptions{})
}

// runServiceAndSidecarForResourceConsumer creates service and runs resource consumer for sidecar container
func runServiceAndSidecarForResourceConsumer(ctx context.Context, c clientset.Interface, ns, name string, kind schema.GroupVersionKind, replicas int, serviceAnnotations map[string]string) {
	ginkgo.By(fmt.Sprintf("Running consuming RC sidecar %s via %s with %v replicas", name, kind, replicas))

	sidecarName := name + "-sidecar"
	serviceSelectors := map[string]string{
		"name": name,
	}
	_, err := createService(ctx, c, sidecarName, ns, serviceAnnotations, serviceSelectors, port, sidecarTargetPort)
	framework.ExpectNoError(err)

	ginkgo.By("Running controller for sidecar")
	controllerName := sidecarName + "-ctrl"
	_, err = createService(ctx, c, controllerName, ns, map[string]string{}, map[string]string{"name": controllerName}, port, targetPort)
	framework.ExpectNoError(err)

	dnsClusterFirst := v1.DNSClusterFirst
	controllerRcConfig := testutils.RCConfig{
		Client:    c,
		Image:     imageutils.GetE2EImage(imageutils.Agnhost),
		Name:      controllerName,
		Namespace: ns,
		Timeout:   timeoutRC,
		Replicas:  1,
		Command:   []string{"/agnhost", "resource-consumer-controller", "--consumer-service-name=" + sidecarName, "--consumer-service-namespace=" + ns, "--consumer-port=80"},
		DNSPolicy: &dnsClusterFirst,
	}

	framework.ExpectNoError(e2erc.RunRC(ctx, controllerRcConfig))
	// Wait for endpoints to propagate for the controller service.
	framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(
		ctx, c, ns, controllerName, 1))
}

func runServiceAndWorkloadForResourceConsumer(ctx context.Context, c clientset.Interface, resourceClient dynamic.ResourceInterface, apiExtensionClient crdclientset.Interface, ns, name string, kind schema.GroupVersionKind, replicas int, cpuLimitMillis, memLimitMb int64, podAnnotations, serviceAnnotations map[string]string, additionalContainers []v1.Container, podResources *v1.ResourceRequirements) {
	ginkgo.By(fmt.Sprintf("Running consuming RC %s via %s with %v replicas", name, kind, replicas))
	_, err := createService(ctx, c, name, ns, serviceAnnotations, map[string]string{"name": name}, port, targetPort)
	framework.ExpectNoError(err)

	rcConfig := testutils.RCConfig{
		Client:               c,
		Image:                imageutils.GetE2EImage(imageutils.ResourceConsumer),
		Name:                 name,
		Namespace:            ns,
		Timeout:              timeoutRC,
		Replicas:             replicas,
		CPURequest:           cpuLimitMillis,
		CPULimit:             cpuLimitMillis,
		MemRequest:           memLimitMb * 1024 * 1024, // MemLimit is in bytes
		MemLimit:             memLimitMb * 1024 * 1024,
		Annotations:          podAnnotations,
		AdditionalContainers: additionalContainers,
	}
	if podResources != nil {
		rcConfig.PodResources = podResources.DeepCopy()
	}

	dpConfig := testutils.DeploymentConfig{
		RCConfig: rcConfig,
	}
	dpConfig.NodeDumpFunc = e2edebug.DumpNodeDebugInfo
	dpConfig.ContainerDumpFunc = e2ekubectl.LogFailedContainers

	switch kind {
	case KindRC:
		framework.ExpectNoError(e2erc.RunRC(ctx, rcConfig))
	case KindDeployment:
		ginkgo.By(fmt.Sprintf("Creating deployment %s in namespace %s", dpConfig.Name, dpConfig.Namespace))
		framework.ExpectNoError(testutils.RunDeployment(ctx, dpConfig))
	case KindReplicaSet:
		rsConfig := testutils.ReplicaSetConfig{
			RCConfig: rcConfig,
		}
		ginkgo.By(fmt.Sprintf("Creating replicaset %s in namespace %s", rsConfig.Name, rsConfig.Namespace))
		framework.ExpectNoError(runReplicaSet(ctx, rsConfig))
	case KindCRD:
		crd := CreateCustomResourceDefinition(ctx, apiExtensionClient)
		crdInstance, err := CreateCustomSubresourceInstance(ctx, ns, name, resourceClient, crd)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("Creating deployment %s backing CRD in namespace %s", dpConfig.Name, dpConfig.Namespace))
		framework.ExpectNoError(testutils.RunDeployment(ctx, dpConfig))

		deployment, err := c.AppsV1().Deployments(dpConfig.Namespace).Get(ctx, dpConfig.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		deployment.SetOwnerReferences([]metav1.OwnerReference{{
			APIVersion: kind.GroupVersion().String(),
			Kind:       crdKind,
			Name:       name,
			UID:        crdInstance.GetUID(),
		}})
		_, err = c.AppsV1().Deployments(dpConfig.Namespace).Update(ctx, deployment, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
	default:
		framework.Failf(invalidKind)
	}

	ginkgo.By(fmt.Sprintf("Running controller"))
	controllerName := name + "-ctrl"
	_, err = createService(ctx, c, controllerName, ns, map[string]string{}, map[string]string{"name": controllerName}, port, targetPort)
	framework.ExpectNoError(err)

	dnsClusterFirst := v1.DNSClusterFirst
	controllerRcConfig := testutils.RCConfig{
		Client:    c,
		Image:     imageutils.GetE2EImage(imageutils.Agnhost),
		Name:      controllerName,
		Namespace: ns,
		Timeout:   timeoutRC,
		Replicas:  1,
		Command:   []string{"/agnhost", "resource-consumer-controller", "--consumer-service-name=" + name, "--consumer-service-namespace=" + ns, "--consumer-port=80"},
		DNSPolicy: &dnsClusterFirst,
	}

	framework.ExpectNoError(e2erc.RunRC(ctx, controllerRcConfig))
	// Wait for endpoints to propagate for the controller service.
	framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(
		ctx, c, ns, controllerName, 1))
}

func CreateHorizontalPodAutoscaler(ctx context.Context, rc *ResourceConsumer, targetRef autoscalingv2.CrossVersionObjectReference, namespace string, metrics []autoscalingv2.MetricSpec, resourceType v1.ResourceName, metricTargetType autoscalingv2.MetricTargetType, metricTargetValue, minReplicas, maxReplicas int32) *autoscalingv2.HorizontalPodAutoscaler {
	hpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      targetRef.Name,
			Namespace: namespace,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: targetRef,
			MinReplicas:    &minReplicas,
			MaxReplicas:    maxReplicas,
			Metrics:        metrics,
		},
	}
	hpa, errHPA := rc.clientSet.AutoscalingV2().HorizontalPodAutoscalers(namespace).Create(ctx, hpa, metav1.CreateOptions{})
	framework.ExpectNoError(errHPA)
	return hpa
}

func CreateResourceHorizontalPodAutoscaler(ctx context.Context, rc *ResourceConsumer, resourceType v1.ResourceName, metricTargetType autoscalingv2.MetricTargetType, metricTargetValue, minReplicas, maxReplicas int32) *autoscalingv2.HorizontalPodAutoscaler {
	targetRef := autoscalingv2.CrossVersionObjectReference{
		APIVersion: rc.kind.GroupVersion().String(),
		Kind:       rc.kind.Kind,
		Name:       rc.name,
	}
	metrics := []autoscalingv2.MetricSpec{
		{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name:   resourceType,
				Target: CreateMetricTargetWithType(resourceType, metricTargetType, metricTargetValue),
			},
		},
	}
	return CreateHorizontalPodAutoscaler(ctx, rc, targetRef, rc.nsName, metrics, resourceType, metricTargetType, metricTargetValue, minReplicas, maxReplicas)
}

func CreateCPUResourceHorizontalPodAutoscaler(ctx context.Context, rc *ResourceConsumer, cpu, minReplicas, maxReplicas int32) *autoscalingv2.HorizontalPodAutoscaler {
	return CreateResourceHorizontalPodAutoscaler(ctx, rc, v1.ResourceCPU, autoscalingv2.UtilizationMetricType, cpu, minReplicas, maxReplicas)
}

// DeleteHorizontalPodAutoscaler delete the horizontalPodAutoscaler for consuming resources.
func DeleteHorizontalPodAutoscaler(ctx context.Context, rc *ResourceConsumer, autoscalerName string) {
	framework.ExpectNoError(rc.clientSet.AutoscalingV1().HorizontalPodAutoscalers(rc.nsName).Delete(ctx, autoscalerName, metav1.DeleteOptions{}))
}

// runReplicaSet launches (and verifies correctness) of a replicaset.
func runReplicaSet(ctx context.Context, config testutils.ReplicaSetConfig) error {
	ginkgo.By(fmt.Sprintf("creating replicaset %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = e2edebug.DumpNodeDebugInfo
	config.ContainerDumpFunc = e2ekubectl.LogFailedContainers
	return testutils.RunReplicaSet(ctx, config)
}

func CreateContainerResourceHorizontalPodAutoscaler(ctx context.Context, rc *ResourceConsumer, resourceType v1.ResourceName, metricTargetType autoscalingv2.MetricTargetType, metricTargetValue, minReplicas, maxReplicas int32) *autoscalingv2.HorizontalPodAutoscaler {
	targetRef := autoscalingv2.CrossVersionObjectReference{
		APIVersion: rc.kind.GroupVersion().String(),
		Kind:       rc.kind.Kind,
		Name:       rc.name,
	}
	metrics := []autoscalingv2.MetricSpec{
		{
			Type: autoscalingv2.ContainerResourceMetricSourceType,
			ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
				Name:      resourceType,
				Container: rc.name,
				Target:    CreateMetricTargetWithType(resourceType, metricTargetType, metricTargetValue),
			},
		},
	}
	return CreateHorizontalPodAutoscaler(ctx, rc, targetRef, rc.nsName, metrics, resourceType, metricTargetType, metricTargetValue, minReplicas, maxReplicas)
}

// DeleteContainerResourceHPA delete the horizontalPodAutoscaler for consuming resources.
func DeleteContainerResourceHPA(ctx context.Context, rc *ResourceConsumer, autoscalerName string) {
	framework.ExpectNoError(rc.clientSet.AutoscalingV2().HorizontalPodAutoscalers(rc.nsName).Delete(ctx, autoscalerName, metav1.DeleteOptions{}))
}

func CreateMetricTargetWithType(resourceType v1.ResourceName, targetType autoscalingv2.MetricTargetType, targetValue int32) autoscalingv2.MetricTarget {
	var metricTarget autoscalingv2.MetricTarget
	if targetType == autoscalingv2.UtilizationMetricType {
		metricTarget = autoscalingv2.MetricTarget{
			Type:               targetType,
			AverageUtilization: &targetValue,
		}
	} else if targetType == autoscalingv2.AverageValueMetricType {
		var averageValue *resource.Quantity
		if resourceType == v1.ResourceCPU {
			averageValue = resource.NewMilliQuantity(int64(targetValue), resource.DecimalSI)
		} else {
			averageValue = resource.NewQuantity(int64(targetValue*megabytes), resource.DecimalSI)
		}
		metricTarget = autoscalingv2.MetricTarget{
			Type:         targetType,
			AverageValue: averageValue,
		}
	}
	return metricTarget
}

func CreateCPUHorizontalPodAutoscalerWithBehavior(ctx context.Context, rc *ResourceConsumer, cpu int32, minReplicas int32, maxRepl int32, behavior *autoscalingv2.HorizontalPodAutoscalerBehavior) *autoscalingv2.HorizontalPodAutoscaler {
	hpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      rc.name,
			Namespace: rc.nsName,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: rc.kind.GroupVersion().String(),
				Kind:       rc.kind.Kind,
				Name:       rc.name,
			},
			MinReplicas: &minReplicas,
			MaxReplicas: maxRepl,
			Metrics: []autoscalingv2.MetricSpec{
				{
					Type: autoscalingv2.ResourceMetricSourceType,
					Resource: &autoscalingv2.ResourceMetricSource{
						Name: v1.ResourceCPU,
						Target: autoscalingv2.MetricTarget{
							Type:               autoscalingv2.UtilizationMetricType,
							AverageUtilization: &cpu,
						},
					},
				},
			},
			Behavior: behavior,
		},
	}
	hpa, errHPA := rc.clientSet.AutoscalingV2().HorizontalPodAutoscalers(rc.nsName).Create(ctx, hpa, metav1.CreateOptions{})
	framework.ExpectNoError(errHPA)
	return hpa
}

func HPABehaviorWithScaleUpAndDownRules(scaleUpRule, scaleDownRule *autoscalingv2.HPAScalingRules) *autoscalingv2.HorizontalPodAutoscalerBehavior {
	return &autoscalingv2.HorizontalPodAutoscalerBehavior{
		ScaleUp:   scaleUpRule,
		ScaleDown: scaleDownRule,
	}
}

func HPABehaviorWithScalingRuleInDirection(scalingDirection ScalingDirection, rule *autoscalingv2.HPAScalingRules) *autoscalingv2.HorizontalPodAutoscalerBehavior {
	var scaleUpRule, scaleDownRule *autoscalingv2.HPAScalingRules
	if scalingDirection == ScaleUpDirection {
		scaleUpRule = rule
	}
	if scalingDirection == ScaleDownDirection {
		scaleDownRule = rule
	}
	return HPABehaviorWithScaleUpAndDownRules(scaleUpRule, scaleDownRule)
}

func HPAScalingRuleWithStabilizationWindow(stabilizationDuration int32) *autoscalingv2.HPAScalingRules {
	return &autoscalingv2.HPAScalingRules{
		StabilizationWindowSeconds: &stabilizationDuration,
	}
}

func HPAScalingRuleWithPolicyDisabled() *autoscalingv2.HPAScalingRules {
	disabledPolicy := autoscalingv2.DisabledPolicySelect
	return &autoscalingv2.HPAScalingRules{
		SelectPolicy: &disabledPolicy,
	}
}

func HPAScalingRuleWithScalingPolicy(policyType autoscalingv2.HPAScalingPolicyType, value, periodSeconds int32) *autoscalingv2.HPAScalingRules {
	stabilizationWindowDisabledDuration := int32(0)
	selectPolicy := autoscalingv2.MaxChangePolicySelect
	return &autoscalingv2.HPAScalingRules{
		Policies: []autoscalingv2.HPAScalingPolicy{
			{
				Type:          policyType,
				Value:         value,
				PeriodSeconds: periodSeconds,
			},
		},
		SelectPolicy:               &selectPolicy,
		StabilizationWindowSeconds: &stabilizationWindowDisabledDuration,
	}
}

func HPAScalingRuleWithToleranceMilli(toleranceMilli int64) *autoscalingv2.HPAScalingRules {
	quantity := resource.NewMilliQuantity(toleranceMilli, resource.DecimalSI)
	return &autoscalingv2.HPAScalingRules{
		Tolerance: quantity,
	}
}

func HPABehaviorWithStabilizationWindows(upscaleStabilization, downscaleStabilization time.Duration) *autoscalingv2.HorizontalPodAutoscalerBehavior {
	scaleUpRule := HPAScalingRuleWithStabilizationWindow(int32(upscaleStabilization.Seconds()))
	scaleDownRule := HPAScalingRuleWithStabilizationWindow(int32(downscaleStabilization.Seconds()))
	return HPABehaviorWithScaleUpAndDownRules(scaleUpRule, scaleDownRule)
}

func HPABehaviorWithScaleDisabled(scalingDirection ScalingDirection) *autoscalingv2.HorizontalPodAutoscalerBehavior {
	scalingRule := HPAScalingRuleWithPolicyDisabled()
	return HPABehaviorWithScalingRuleInDirection(scalingDirection, scalingRule)
}

func HPABehaviorWithScaleLimitedByNumberOfPods(scalingDirection ScalingDirection, numberOfPods, periodSeconds int32) *autoscalingv2.HorizontalPodAutoscalerBehavior {
	scalingRule := HPAScalingRuleWithScalingPolicy(autoscalingv2.PodsScalingPolicy, numberOfPods, periodSeconds)
	return HPABehaviorWithScalingRuleInDirection(scalingDirection, scalingRule)
}

func HPABehaviorWithScaleLimitedByPercentage(scalingDirection ScalingDirection, percentage, periodSeconds int32) *autoscalingv2.HorizontalPodAutoscalerBehavior {
	scalingRule := HPAScalingRuleWithScalingPolicy(autoscalingv2.PercentScalingPolicy, percentage, periodSeconds)
	return HPABehaviorWithScalingRuleInDirection(scalingDirection, scalingRule)
}

func DeleteHPAWithBehavior(ctx context.Context, rc *ResourceConsumer, autoscalerName string) {
	framework.ExpectNoError(rc.clientSet.AutoscalingV2().HorizontalPodAutoscalers(rc.nsName).Delete(ctx, autoscalerName, metav1.DeleteOptions{}))
}

// SidecarStatusType type for sidecar status
type SidecarStatusType bool

const (
	Enable  SidecarStatusType = true
	Disable SidecarStatusType = false
)

// SidecarWorkloadType type of the sidecar
type SidecarWorkloadType string

const (
	Busy SidecarWorkloadType = "Busy"
	Idle SidecarWorkloadType = "Idle"
)

func CreateCustomResourceDefinition(ctx context.Context, c crdclientset.Interface) *apiextensionsv1.CustomResourceDefinition {
	crdSchema := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: crdNamePlural + "." + crdGroup},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: crdGroup,
			Scope: apiextensionsv1.ResourceScope("Namespaced"),
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   crdNamePlural,
				Singular: crdName,
				Kind:     crdKind,
				ListKind: "TestCRDList",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name:    crdVersion,
				Served:  true,
				Storage: true,
				Schema:  fixtures.AllowAllSchema(),
				Subresources: &apiextensionsv1.CustomResourceSubresources{
					Scale: &apiextensionsv1.CustomResourceSubresourceScale{
						SpecReplicasPath:   ".spec.replicas",
						StatusReplicasPath: ".status.replicas",
						LabelSelectorPath:  ptr.To(".status.selector"),
					},
				},
			}},
		},
		Status: apiextensionsv1.CustomResourceDefinitionStatus{},
	}
	// Create Custom Resource Definition if it's not present.
	crd, err := c.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, crdSchema.Name, metav1.GetOptions{})
	if err != nil {
		crd, err = c.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, crdSchema, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		// Wait until just created CRD appears in discovery.
		err = framework.Gomega().Eventually(ctx, framework.RetryNotFound(framework.HandleRetry(func(ctx context.Context) (*metav1.APIResourceList, error) {
			return c.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + "v1")
		}))).Should(framework.MakeMatcher(func(actual *metav1.APIResourceList) (func() string, error) {
			for _, g := range actual.APIResources {
				if g.Name == crd.Spec.Names.Plural {
					return nil, nil
				}
			}
			return func() string {
				return fmt.Sprintf("CRD %s not found in discovery", crd.Spec.Names.Plural)
			}, nil
		}))
		framework.ExpectNoError(err)
		ginkgo.By(fmt.Sprintf("Successfully created Custom Resource Definition: %v", crd))
	}
	return crd
}

func CreateCustomSubresourceInstance(ctx context.Context, namespace, name string, client dynamic.ResourceInterface, definition *apiextensionsv1.CustomResourceDefinition) (*unstructured.Unstructured, error) {
	instance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": crdGroup + "/" + crdVersion,
			"kind":       crdKind,
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"spec": map[string]interface{}{
				"num":      int64(1),
				"replicas": int64(1),
			},
			"status": map[string]interface{}{
				"replicas": int64(1),
				"selector": "name=" + name,
			},
		},
	}
	instance, err := client.Create(ctx, instance, metav1.CreateOptions{})
	if err != nil {
		framework.Logf("%#v", instance)
		return nil, err
	}
	createdObjectMeta, err := meta.Accessor(instance)
	if err != nil {
		return nil, fmt.Errorf("Error while creating object meta: %w", err)
	}
	if len(createdObjectMeta.GetUID()) == 0 {
		return nil, fmt.Errorf("Missing UUID: %v", instance)
	}
	ginkgo.By(fmt.Sprintf("Successfully created instance of CRD of kind %v: %v", definition.Kind, instance))
	return instance, nil
}
