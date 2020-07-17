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

package metrics

import (
	"context"
	"fmt"
	"regexp"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"k8s.io/klog/v2"
)

const (
	// insecureSchedulerPort is the default port for the scheduler status server.
	// May be overridden by a flag at startup.
	// Deprecated: use the secure KubeSchedulerPort instead.
	insecureSchedulerPort = 10251
	// insecureKubeControllerManagerPort is the default port for the controller manager status server.
	// May be overridden by a flag at startup.
	// Deprecated: use the secure KubeControllerManagerPort instead.
	insecureKubeControllerManagerPort = 10252
)

// Collection is metrics collection of components
type Collection struct {
	APIServerMetrics         APIServerMetrics
	ControllerManagerMetrics ControllerManagerMetrics
	KubeletMetrics           map[string]KubeletMetrics
	SchedulerMetrics         SchedulerMetrics
	ClusterAutoscalerMetrics ClusterAutoscalerMetrics
}

// Grabber provides functions which grab metrics from components
type Grabber struct {
	client                            clientset.Interface
	externalClient                    clientset.Interface
	grabFromAPIServer                 bool
	grabFromControllerManager         bool
	grabFromKubelets                  bool
	grabFromScheduler                 bool
	grabFromClusterAutoscaler         bool
	kubeScheduler                     string
	kubeControllerManager             string
	waitForControllerManagerReadyOnce sync.Once
}

// NewMetricsGrabber returns new metrics which are initialized.
func NewMetricsGrabber(c clientset.Interface, ec clientset.Interface, kubelets bool, scheduler bool, controllers bool, apiServer bool, clusterAutoscaler bool) (*Grabber, error) {

	kubeScheduler := ""
	kubeControllerManager := ""

	regKubeScheduler := regexp.MustCompile("kube-scheduler-.*")
	regKubeControllerManager := regexp.MustCompile("kube-controller-manager-.*")

	podList, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	if len(podList.Items) < 1 {
		klog.Warningf("Can't find any pods in namespace %s to grab metrics from", metav1.NamespaceSystem)
	}
	for _, pod := range podList.Items {
		if regKubeScheduler.MatchString(pod.Name) {
			kubeScheduler = pod.Name
		}
		if regKubeControllerManager.MatchString(pod.Name) {
			kubeControllerManager = pod.Name
		}
		if kubeScheduler != "" && kubeControllerManager != "" {
			break
		}
	}
	if kubeScheduler == "" {
		scheduler = false
		klog.Warningf("Can't find kube-scheduler pod. Grabbing metrics from kube-scheduler is disabled.")
	}
	if kubeControllerManager == "" {
		controllers = false
		klog.Warningf("Can't find kube-controller-manager pod. Grabbing metrics from kube-controller-manager is disabled.")
	}
	if ec == nil {
		klog.Warningf("Did not receive an external client interface. Grabbing metrics from ClusterAutoscaler is disabled.")
	}

	return &Grabber{
		client:                    c,
		externalClient:            ec,
		grabFromAPIServer:         apiServer,
		grabFromControllerManager: controllers,
		grabFromKubelets:          kubelets,
		grabFromScheduler:         scheduler,
		grabFromClusterAutoscaler: clusterAutoscaler,
		kubeScheduler:             kubeScheduler,
		kubeControllerManager:     kubeControllerManager,
	}, nil
}

// HasControlPlanePods returns true if metrics grabber was able to find control-plane pods
func (g *Grabber) HasControlPlanePods() bool {
	return g.kubeScheduler != "" && g.kubeControllerManager != ""
}

// GrabFromKubelet returns metrics from kubelet
func (g *Grabber) GrabFromKubelet(nodeName string) (KubeletMetrics, error) {
	nodes, err := g.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{FieldSelector: fields.Set{"metadata.name": nodeName}.AsSelector().String()})
	if err != nil {
		return KubeletMetrics{}, err
	}
	if len(nodes.Items) != 1 {
		return KubeletMetrics{}, fmt.Errorf("Error listing nodes with name %v, got %v", nodeName, nodes.Items)
	}
	kubeletPort := nodes.Items[0].Status.DaemonEndpoints.KubeletEndpoint.Port
	return g.grabFromKubeletInternal(nodeName, int(kubeletPort))
}

func (g *Grabber) grabFromKubeletInternal(nodeName string, kubeletPort int) (KubeletMetrics, error) {
	if kubeletPort <= 0 || kubeletPort > 65535 {
		return KubeletMetrics{}, fmt.Errorf("Invalid Kubelet port %v. Skipping Kubelet's metrics gathering", kubeletPort)
	}
	output, err := g.getMetricsFromNode(nodeName, int(kubeletPort))
	if err != nil {
		return KubeletMetrics{}, err
	}
	return parseKubeletMetrics(output)
}

// GrabFromScheduler returns metrics from scheduler
func (g *Grabber) GrabFromScheduler() (SchedulerMetrics, error) {
	if g.kubeScheduler == "" {
		return SchedulerMetrics{}, fmt.Errorf("kube-scheduler pod is not registered. Skipping Scheduler's metrics gathering")
	}
	output, err := g.getMetricsFromPod(g.client, g.kubeScheduler, metav1.NamespaceSystem, insecureSchedulerPort)
	if err != nil {
		return SchedulerMetrics{}, err
	}
	return parseSchedulerMetrics(output)
}

// GrabFromClusterAutoscaler returns metrics from cluster autoscaler
func (g *Grabber) GrabFromClusterAutoscaler() (ClusterAutoscalerMetrics, error) {
	if !g.HasControlPlanePods() && g.externalClient == nil {
		return ClusterAutoscalerMetrics{}, fmt.Errorf("Did not find control-plane pods. Skipping ClusterAutoscaler's metrics gathering")
	}
	var client clientset.Interface
	var namespace string
	if g.externalClient != nil {
		client = g.externalClient
		namespace = "kubemark"
	} else {
		client = g.client
		namespace = metav1.NamespaceSystem
	}
	output, err := g.getMetricsFromPod(client, "cluster-autoscaler", namespace, 8085)
	if err != nil {
		return ClusterAutoscalerMetrics{}, err
	}
	return parseClusterAutoscalerMetrics(output)
}

// GrabFromControllerManager returns metrics from controller manager
func (g *Grabber) GrabFromControllerManager() (ControllerManagerMetrics, error) {
	if g.kubeControllerManager == "" {
		return ControllerManagerMetrics{}, fmt.Errorf("kube-controller-manager pod is not registered. Skipping ControllerManager's metrics gathering")
	}

	var err error
	podName := g.kubeControllerManager
	g.waitForControllerManagerReadyOnce.Do(func() {
		if readyErr := e2epod.WaitForPodsReady(g.client, metav1.NamespaceSystem, podName, 0); readyErr != nil {
			err = fmt.Errorf("error waiting for controller manager pod to be ready: %w", readyErr)
			return
		}

		var lastMetricsFetchErr error
		if metricsWaitErr := wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
			_, lastMetricsFetchErr = g.getMetricsFromPod(g.client, podName, metav1.NamespaceSystem, insecureKubeControllerManagerPort)
			return lastMetricsFetchErr == nil, nil
		}); metricsWaitErr != nil {
			err = fmt.Errorf("error waiting for controller manager pod to expose metrics: %v; %v", metricsWaitErr, lastMetricsFetchErr)
			return
		}
	})
	if err != nil {
		return ControllerManagerMetrics{}, err
	}

	output, err := g.getMetricsFromPod(g.client, podName, metav1.NamespaceSystem, insecureKubeControllerManagerPort)
	if err != nil {
		return ControllerManagerMetrics{}, err
	}
	return parseControllerManagerMetrics(output)
}

// GrabFromAPIServer returns metrics from API server
func (g *Grabber) GrabFromAPIServer() (APIServerMetrics, error) {
	output, err := g.getMetricsFromAPIServer()
	if err != nil {
		return APIServerMetrics{}, nil
	}
	return parseAPIServerMetrics(output)
}

// Grab returns metrics from corresponding component
func (g *Grabber) Grab() (Collection, error) {
	result := Collection{}
	var errs []error
	if g.grabFromAPIServer {
		metrics, err := g.GrabFromAPIServer()
		if err != nil {
			errs = append(errs, err)
		} else {
			result.APIServerMetrics = metrics
		}
	}
	if g.grabFromScheduler {
		metrics, err := g.GrabFromScheduler()
		if err != nil {
			errs = append(errs, err)
		} else {
			result.SchedulerMetrics = metrics
		}
	}
	if g.grabFromControllerManager {
		metrics, err := g.GrabFromControllerManager()
		if err != nil {
			errs = append(errs, err)
		} else {
			result.ControllerManagerMetrics = metrics
		}
	}
	if g.grabFromClusterAutoscaler {
		metrics, err := g.GrabFromClusterAutoscaler()
		if err != nil {
			errs = append(errs, err)
		} else {
			result.ClusterAutoscalerMetrics = metrics
		}
	}
	if g.grabFromKubelets {
		result.KubeletMetrics = make(map[string]KubeletMetrics)
		nodes, err := g.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			errs = append(errs, err)
		} else {
			for _, node := range nodes.Items {
				kubeletPort := node.Status.DaemonEndpoints.KubeletEndpoint.Port
				metrics, err := g.grabFromKubeletInternal(node.Name, int(kubeletPort))
				if err != nil {
					errs = append(errs, err)
				}
				result.KubeletMetrics[node.Name] = metrics
			}
		}
	}
	if len(errs) > 0 {
		return result, fmt.Errorf("Errors while grabbing metrics: %v", errs)
	}
	return result, nil
}

func (g *Grabber) getMetricsFromPod(client clientset.Interface, podName string, namespace string, port int) (string, error) {
	rawOutput, err := client.CoreV1().RESTClient().Get().
		Namespace(namespace).
		Resource("pods").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", podName, port)).
		Suffix("metrics").
		Do(context.TODO()).Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
