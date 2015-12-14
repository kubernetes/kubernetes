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

package metrics

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

type MetricsCollection struct {
	ApiServerMetrics         ApiServerMetrics
	ControllerManagerMetrics ControllerManagerMetrics
	KubeletMetrics           map[string]KubeletMetrics
	SchedulerMetrics         SchedulerMetrics
}

type MetricsGrabber struct {
	client                    *client.Client
	grabFromApiServer         bool
	grabFromControllerManager bool
	grabFromKubelets          bool
	grabFromScheduler         bool
	kubeletPort               int
	masterName                string
	registeredMaster          bool
}

// TODO: find a better way of figuring out if given node is a registered master.
func isMasterNode(node *api.Node) bool {
	return strings.HasSuffix(node.Name, "master")
}

func NewMetricsGrabber(c *client.Client, kubelets bool, scheduler bool, controllers bool, apiServer bool) (*MetricsGrabber, error) {
	kubeletPort := 0
	registeredMaster := false
	masterName := ""
	nodeList, err := c.Nodes().List(api.ListOptions{})
	if err != nil {
		return nil, err
	}
	if len(nodeList.Items) < 1 {
		glog.Warning("Can't find any Nodes in the API server to grab metrics from")
	}
	for _, node := range nodeList.Items {
		if isMasterNode(&node) {
			registeredMaster = true
			masterName = node.Name
		} else {
			// We assume that all Kubelets run on the same port, except possibly Master's Kubelet.
			// This will need to get change if assumption is prover wrong.
			kubeletPort = node.Status.DaemonEndpoints.KubeletEndpoint.Port
		}
	}
	if !registeredMaster {
		scheduler = false
		controllers = false
		glog.Warningf("Master node is not registered. Grabbing metrics from Scheduler and ControllerManager is disabled.")
	}

	return &MetricsGrabber{
		client:                    c,
		grabFromApiServer:         apiServer,
		grabFromControllerManager: controllers,
		grabFromKubelets:          kubelets,
		grabFromScheduler:         scheduler,
		kubeletPort:               kubeletPort,
		masterName:                masterName,
		registeredMaster:          registeredMaster,
	}, nil
}

func (g *MetricsGrabber) GrabFromKubelet(nodeName string, unknownMetrics sets.String) (KubeletMetrics, error) {
	if g.kubeletPort == 0 {
		return KubeletMetrics{}, fmt.Errorf("MetricsGrabber wasn't able to find Kubelet port during startup. Skipping Kubelet's metrics gathering.")
	}
	output, err := g.getMetricsFromNode(nodeName)
	if err != nil {
		return KubeletMetrics{}, err
	}
	return parseKubeletMetrics(output, unknownMetrics)
}

func (g *MetricsGrabber) GrabFromScheduler(unknownMetrics sets.String) (SchedulerMetrics, error) {
	if !g.registeredMaster {
		return SchedulerMetrics{}, fmt.Errorf("Master's Kubelet is not registered. Skipping Scheduler's metrics gathering.")
	}
	output, err := g.getMetricsFromPod(fmt.Sprintf("%v-%v", "kube-scheduler", g.masterName), api.NamespaceSystem, ports.SchedulerPort)
	if err != nil {
		return SchedulerMetrics{}, err
	}
	return parseSchedulerMetrics(output, unknownMetrics)
}

func (g *MetricsGrabber) GrabFromControllerManager(unknownMetrics sets.String) (ControllerManagerMetrics, error) {
	if !g.registeredMaster {
		return ControllerManagerMetrics{}, fmt.Errorf("Master's Kubelet is not registered. Skipping ControllerManager's metrics gathering.")
	}
	output, err := g.getMetricsFromPod(fmt.Sprintf("%v-%v", "kube-controller-manager", g.masterName), api.NamespaceSystem, ports.ControllerManagerPort)
	if err != nil {
		return ControllerManagerMetrics{}, err
	}
	return parseControllerManagerMetrics(output, unknownMetrics)
}

func (g *MetricsGrabber) GrabFromApiServer(unknownMetrics sets.String) (ApiServerMetrics, error) {
	output, err := g.getMetricsFromApiServer()
	if err != nil {
		return ApiServerMetrics{}, nil
	}
	return parseApiServerMetrics(output, unknownMetrics)
}

func (g *MetricsGrabber) Grab(unknownMetrics sets.String) (MetricsCollection, error) {
	return MetricsCollection{}, nil
}
