/*
Copyright 2015 Google Inc. All rights reserved.

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

/*
This soak tests places a specified number of pods on each node and then
repeatedly sends queries to a service running on these pods via
a serivce
*/

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	queriesAverage = flag.Int("queries", 10, "Number of hostname queries to make in each iteration per pod on average")
	podsPerNode    = flag.Int("pods_per_node", 1, "Number of serve_hostname pods per node")
	upTo           = flag.Int("up_to", 1, "Number of iterations or -1 for no limit")
)

const (
	deleteTimeout        = 2 * time.Minute
	endpointTimeout      = 5 * time.Minute
	podCreateTimeout     = 2 * time.Minute
	podStartTimeout      = 30 * time.Minute
	serviceCreateTimeout = 2 * time.Minute
)

func main() {
	flag.Parse()

	glog.Infof("Starting serve_hostnames soak test with queries=%d and podsPerNode=%d upTo=%d",
		*queriesAverage, *podsPerNode, *upTo)

	settings, err := clientcmd.LoadFromFile(filepath.Join(os.Getenv("HOME"), ".kube", ".kubeconfig"))
	if err != nil {
		glog.Fatalf("Error loading configuration: %v", err.Error())
	}
	config, err := clientcmd.NewDefaultClientConfig(*settings, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		glog.Fatalf("Failed to construct config: %v", err)
	}

	c, err := client.New(config)
	if err != nil {
		glog.Fatalf("Failed to make client: %v", err)
	}

	nodes, err := c.Nodes().List()
	if err != nil {
		glog.Fatalf("Failed to list nodes: %v", err)
	}

	if len(nodes.Items) == 0 {
		glog.Fatalf("Failed to find any nodes.")
	}

	glog.Infof("Nodes found on this cluster:")
	for i, node := range nodes.Items {
		glog.Infof("%d: %s", i, node.Name)
	}

	queries := *queriesAverage * len(nodes.Items) * *podsPerNode

	// Make a unique namespace for this test.
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	ns := "serve-hostnames-" + strconv.Itoa(r.Int()%10000)
	glog.Infof("Using namespace %s for this test.", ns)

	// Create a service for these pods.
	glog.Infof("Creating service %s/serve-hostnames", ns)
	// Make several attempts to create a service.
	var svc *api.Service
	for start := time.Now(); time.Since(start) < serviceCreateTimeout; time.Sleep(2 * time.Second) {
		t := time.Now()
		svc, err = c.Services(ns).Create(&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: "serve-hostnames",
				Labels: map[string]string{
					"name": "serve-hostname",
				},
			},
			Spec: api.ServiceSpec{
				Port:       9376,
				TargetPort: util.NewIntOrStringFromInt(9376),
				Selector: map[string]string{
					"name": "serve-hostname",
				},
			},
		})
		glog.V(4).Infof("Service create %s/server-hostnames took %v", ns, time.Since(t))
		if err == nil {
			break
		}
		glog.Warningf("After %v failed to create service %s/serve-hostnames: %v", time.Since(start), ns, err)
	}
	if err != nil {
		glog.Warningf("Unable to create service %s/%s: %v", ns, svc.Name, err)
		return
	}
	// Clean up service
	defer func() {
		glog.Infof("Cleaning up service %s/server-hostnames", ns)
		// Make several attempts to delete the service.
		for start := time.Now(); time.Since(start) < deleteTimeout; time.Sleep(1 * time.Second) {
			if err := c.Services(ns).Delete(svc.Name); err == nil {
				return
			}
			glog.Warningf("After %v unable to delete service %s/%s: %v", time.Since(start), ns, svc.Name, err)
		}
	}()

	// Put serve-hostname pods on each node.
	podNames := []string{}
	for i, node := range nodes.Items {
		for j := 0; j < *podsPerNode; j++ {
			podName := fmt.Sprintf("serve-hostname-%d-%d", i, j)
			podNames = append(podNames, podName)
			// Make several attempts
			for start := time.Now(); time.Since(start) < podCreateTimeout; time.Sleep(2 * time.Second) {
				glog.Infof("Creating pod %s/%s on node %s", ns, podName, node.Name)
				t := time.Now()
				_, err = c.Pods(ns).Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
						Labels: map[string]string{
							"name": "serve-hostname",
						},
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Name:  "serve-hostname",
								Image: "kubernetes/serve_hostname:1.1",
								Ports: []api.ContainerPort{{ContainerPort: 9376}},
							},
						},
						Host: node.Name,
					},
				})
				glog.V(4).Infof("Pod create %s/%s request took %v", ns, podName, time.Since(t))
				if err == nil {
					break
				}
				glog.Warningf("After %s failed to create pod %s/%s: %v", time.Since(start), ns, podName, err)
			}
			if err != nil {
				glog.Warningf("Failed to create pod %s/%s: %v", ns, podName, err)
				return
			}
		}
	}
	// Clean up the pods
	defer func() {
		glog.Info("Cleaning up pods")
		// Make several attempts to delete the pods.
		for _, podName := range podNames {
			for start := time.Now(); time.Since(start) < deleteTimeout; time.Sleep(1 * time.Second) {
				if err = c.Pods(ns).Delete(podName); err == nil {
					break
				}
				glog.Warningf("After %v failed to delete pod %s/%s: %v", time.Since(start), ns, podName, err)
			}
		}
	}()

	glog.Info("Waiting for the serve-hostname pods to be ready")
	for _, podName := range podNames {
		var pod *api.Pod
		for start := time.Now(); time.Since(start) < podStartTimeout; time.Sleep(5 * time.Second) {
			pod, err = c.Pods(ns).Get(podName)
			if err != nil {
				glog.Infof("Get pod %s/%s failed, ignoring for %v: %v", ns, podName, err, podStartTimeout)
				continue
			}
			if pod.Status.Phase == api.PodRunning {
				break
			}
		}
		if pod.Status.Phase != api.PodRunning {
			glog.Warningf("Gave up waiting on pod %s/%s to be running (saw %v)", ns, podName, pod.Status.Phase)
			return
		}
		glog.Infof("%s/%s is running", ns, podName)
	}

	// Wait for the endpoints to propagate.
	for start := time.Now(); time.Since(start) < endpointTimeout; time.Sleep(10 * time.Second) {
		hostname, err := c.Get().
			Namespace(ns).
			Prefix("proxy").
			Resource("services").
			Name("serve-hostnames").
			DoRaw()
		if err != nil {
			glog.Infof("After %v while making a proxy call got error %v", time.Since(start), err)
			continue
		}
		var r api.Status
		if err := json.Unmarshal(hostname, &r); err != nil {
			break
		}
		if r.Status == api.StatusFailure {
			glog.Infof("After %v got status %v", time.Since(start), string(hostname))
			continue
		}
		break
	}

	// Repeatedly make requests.
	for iteration := 0; iteration != *upTo; iteration++ {
		responses := make(map[string]int, *podsPerNode*len(nodes.Items))
		start := time.Now()
		for q := 0; q < queries; q++ {
			t := time.Now()
			hostname, err := c.Get().
				Namespace(ns).
				Prefix("proxy").
				Resource("services").
				Name("serve-hostnames").
				DoRaw()
			glog.V(4).Infof("Proxy call in namespace %s took %v", ns, time.Since(t))
			if err != nil {
				glog.Infof("Call failed during iteration %d query %d : %v", iteration, q, err)
			} else {
				responses[string(hostname)]++
			}

		}
		for k, v := range responses {
			glog.V(4).Infof("%s: %d  ", k, v)
		}
		// Report any nodes that did not respond.
		for n, node := range nodes.Items {
			for i := 0; i < *podsPerNode; i++ {
				name := fmt.Sprintf("serve-hostname-%d-%d", n, i)
				if _, ok := responses[name]; !ok {
					glog.Warningf("No response from pod %s on node %s at iteration %d", name, node.Name, iteration)
				}
			}
		}
		glog.Infof("Iteration %d took %v for %d queries (%.2f QPS)", iteration, time.Since(start), queries, float64(queries)/time.Since(start).Seconds())
	}
}
