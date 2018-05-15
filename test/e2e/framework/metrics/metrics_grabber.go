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
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util/system"

	"github.com/golang/glog"
)

const (
	ProxyTimeout = 2 * time.Minute
)

type MetricsCollection struct {
	ApiServerMetrics         ApiServerMetrics
	ControllerManagerMetrics ControllerManagerMetrics
	KubeletMetrics           map[string]KubeletMetrics
	SchedulerMetrics         SchedulerMetrics
	ClusterAutoscalerMetrics ClusterAutoscalerMetrics
}

type MetricsGrabber struct {
	client                    clientset.Interface
	externalClient            clientset.Interface
	grabFromApiServer         bool
	grabFromControllerManager bool
	grabFromKubelets          bool
	grabFromScheduler         bool
	grabFromClusterAutoscaler bool
	masterName                string
	registeredMaster          bool
}

func NewMetricsGrabber(c clientset.Interface, ec clientset.Interface, kubelets bool, scheduler bool, controllers bool, apiServer bool, clusterAutoscaler bool) (*MetricsGrabber, error) {
	registeredMaster := false
	masterName := ""
	nodeList, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	if len(nodeList.Items) < 1 {
		glog.Warning("Can't find any Nodes in the API server to grab metrics from")
	}
	for _, node := range nodeList.Items {
		if system.IsMasterNode(node.Name) {
			registeredMaster = true
			masterName = node.Name
			break
		}
	}
	if !registeredMaster {
		scheduler = false
		controllers = false
		clusterAutoscaler = ec != nil
		if clusterAutoscaler {
			glog.Warningf("Master node is not registered. Grabbing metrics from Scheduler, ControllerManager is disabled.")
		} else {
			glog.Warningf("Master node is not registered. Grabbing metrics from Scheduler, ControllerManager and ClusterAutoscaler is disabled.")
		}
	}

	return &MetricsGrabber{
		client:                    c,
		externalClient:            ec,
		grabFromApiServer:         apiServer,
		grabFromControllerManager: controllers,
		grabFromKubelets:          kubelets,
		grabFromScheduler:         scheduler,
		grabFromClusterAutoscaler: clusterAutoscaler,
		masterName:                masterName,
		registeredMaster:          registeredMaster,
	}, nil
}

// HasRegisteredMaster returns if metrics grabber was able to find a master node
func (g *MetricsGrabber) HasRegisteredMaster() bool {
	return g.registeredMaster
}

func (g *MetricsGrabber) GrabFromKubelet(nodeName string) (KubeletMetrics, error) {
	nodes, err := g.client.CoreV1().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{api.ObjectNameField: nodeName}.AsSelector().String()})
	if err != nil {
		return KubeletMetrics{}, err
	}
	if len(nodes.Items) != 1 {
		return KubeletMetrics{}, fmt.Errorf("Error listing nodes with name %v, got %v", nodeName, nodes.Items)
	}
	kubeletPort := nodes.Items[0].Status.DaemonEndpoints.KubeletEndpoint.Port
	return g.grabFromKubeletInternal(nodeName, int(kubeletPort))
}

func (g *MetricsGrabber) grabFromKubeletInternal(nodeName string, kubeletPort int) (KubeletMetrics, error) {
	if kubeletPort <= 0 || kubeletPort > 65535 {
		return KubeletMetrics{}, fmt.Errorf("Invalid Kubelet port %v. Skipping Kubelet's metrics gathering.", kubeletPort)
	}
	output, err := g.getMetricsFromNode(nodeName, int(kubeletPort))
	if err != nil {
		return KubeletMetrics{}, err
	}
	return parseKubeletMetrics(output)
}

func (g *MetricsGrabber) GrabFromScheduler() (SchedulerMetrics, error) {
	if !g.registeredMaster {
		return SchedulerMetrics{}, fmt.Errorf("Master's Kubelet is not registered. Skipping Scheduler's metrics gathering.")
	}
	output, err := g.getMetricsFromPod(g.client, fmt.Sprintf("%v-%v", "kube-scheduler", g.masterName), metav1.NamespaceSystem, ports.SchedulerPort)
	if err != nil {
		return SchedulerMetrics{}, err
	}
	return parseSchedulerMetrics(output)
}

func (g *MetricsGrabber) GrabFromClusterAutoscaler() (ClusterAutoscalerMetrics, error) {
	if !g.registeredMaster && g.externalClient == nil {
		return ClusterAutoscalerMetrics{}, fmt.Errorf("Master's Kubelet is not registered. Skipping ClusterAutoscaler's metrics gathering.")
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

func (g *MetricsGrabber) GrabFromControllerManager() (ControllerManagerMetrics, error) {
	if !g.registeredMaster {
		return ControllerManagerMetrics{}, fmt.Errorf("Master's Kubelet is not registered. Skipping ControllerManager's metrics gathering.")
	}
	output, err := g.getMetricsFromPod(g.client, fmt.Sprintf("%v-%v", "kube-controller-manager", g.masterName), metav1.NamespaceSystem, ports.InsecureKubeControllerManagerPort)
	if err != nil {
		return ControllerManagerMetrics{}, err
	}
	return parseControllerManagerMetrics(output)
}

func (g *MetricsGrabber) GrabFromApiServer() (ApiServerMetrics, error) {
	output, err := g.getMetricsFromApiServer()
	if err != nil {
		return ApiServerMetrics{}, nil
	}
	return parseApiServerMetrics(output)
}

func (g *MetricsGrabber) Grab() (MetricsCollection, error) {
	result := MetricsCollection{}
	var errs []error
	if g.grabFromApiServer {
		metrics, err := g.GrabFromApiServer()
		if err != nil {
			errs = append(errs, err)
		} else {
			result.ApiServerMetrics = metrics
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
		nodes, err := g.client.CoreV1().Nodes().List(metav1.ListOptions{})
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

func (g *MetricsGrabber) getMetricsFromPod(client clientset.Interface, podName string, namespace string, port int) (string, error) {
	rawOutput, err := client.CoreV1().RESTClient().Get().
		Namespace(namespace).
		Resource("pods").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", podName, port)).
		Suffix("metrics").
		Do().Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
