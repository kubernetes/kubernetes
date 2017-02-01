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

/*
This soak tests places a specified number of pods on each node and then
repeatedly sends queries to a service running on these pods via
a service.
*/

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

var (
	queriesAverage = flag.Int("queries", 100, "Number of hostname queries to make in each iteration per pod on average")
	podsPerNode    = flag.Int("pods_per_node", 1, "Number of serve_hostname pods per node")
	upTo           = flag.Int("up_to", 1, "Number of iterations or -1 for no limit")
	maxPar         = flag.Int("max_in_flight", 100, "Maximum number of queries in flight")
)

const (
	deleteTimeout          = 2 * time.Minute
	endpointTimeout        = 5 * time.Minute
	nodeListTimeout        = 2 * time.Minute
	podCreateTimeout       = 2 * time.Minute
	podStartTimeout        = 30 * time.Minute
	serviceCreateTimeout   = 2 * time.Minute
	namespaceDeleteTimeout = 5 * time.Minute
)

func main() {
	flag.Parse()

	glog.Infof("Starting cauldron soak test with queries=%d podsPerNode=%d upTo=%d maxPar=%d",
		*queriesAverage, *podsPerNode, *upTo, *maxPar)

	cc, err := restclient.InClusterConfig()
	if err != nil {
		glog.Fatalf("Failed to make client: %v", err)
	}

	client, err := clientset.NewForConfig(cc)
	if err != nil {
		glog.Fatalf("Failed to make client: %v", err)
	}

	var nodes *api.NodeList
	for start := time.Now(); time.Since(start) < nodeListTimeout; time.Sleep(2 * time.Second) {
		nodes, err = client.Core().Nodes().List(metav1.ListOptions{})
		if err == nil {
			break
		}
		glog.Warningf("Failed to list nodes: %v", err)
	}
	if err != nil {
		glog.Fatalf("Giving up trying to list nodes: %v", err)
	}

	if len(nodes.Items) == 0 {
		glog.Fatalf("Failed to find any nodes.")
	}

	glog.Infof("Found %d nodes on this cluster:", len(nodes.Items))
	for i, node := range nodes.Items {
		glog.Infof("%d: %s", i, node.Name)
	}

	queries := *queriesAverage * len(nodes.Items) * *podsPerNode

	// Create a uniquely named namespace.
	got, err := client.Core().Namespaces().Create(&api.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: "serve-hostnames-"}})
	if err != nil {
		glog.Fatalf("Failed to create namespace: %v", err)
	}
	ns := got.Name
	defer func(ns string) {
		if err := client.Core().Namespaces().Delete(ns, nil); err != nil {
			glog.Warningf("Failed to delete namespace ns: %e", ns, err)
		} else {
			// wait until the namespace disappears
			for i := 0; i < int(namespaceDeleteTimeout/time.Second); i++ {
				if _, err := client.Core().Namespaces().Get(ns, metav1.GetOptions{}); err != nil {
					if errors.IsNotFound(err) {
						return
					}
				}
				time.Sleep(time.Second)
			}
		}
	}(ns)
	glog.Infof("Created namespace %s", ns)

	// Create a service for these pods.
	glog.Infof("Creating service %s/serve-hostnames", ns)
	// Make several attempts to create a service.
	var svc *api.Service
	for start := time.Now(); time.Since(start) < serviceCreateTimeout; time.Sleep(2 * time.Second) {
		t := time.Now()
		svc, err = client.Core().Services(ns).Create(&api.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "serve-hostnames",
				Labels: map[string]string{
					"name": "serve-hostname",
				},
			},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol:   "TCP",
					Port:       9376,
					TargetPort: intstr.FromInt(9376),
				}},
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
		glog.Infof("Cleaning up service %s/serve-hostnames", ns)
		// Make several attempts to delete the service.
		for start := time.Now(); time.Since(start) < deleteTimeout; time.Sleep(1 * time.Second) {
			if err := client.Core().Services(ns).Delete(svc.Name, nil); err == nil {
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
				_, err = client.Core().Pods(ns).Create(&api.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
						Labels: map[string]string{
							"name": "serve-hostname",
						},
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Name:  "serve-hostname",
								Image: "gcr.io/google_containers/serve_hostname:v1.4",
								Ports: []api.ContainerPort{{ContainerPort: 9376}},
							},
						},
						NodeName: node.Name,
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
				if err = client.Core().Pods(ns).Delete(podName, nil); err == nil {
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
			pod, err = client.Core().Pods(ns).Get(podName, metav1.GetOptions{})
			if err != nil {
				glog.Warningf("Get pod %s/%s failed, ignoring for %v: %v", ns, podName, err, podStartTimeout)
				continue
			}
			if pod.Status.Phase == api.PodRunning {
				break
			}
		}
		if pod.Status.Phase != api.PodRunning {
			glog.Warningf("Gave up waiting on pod %s/%s to be running (saw %v)", ns, podName, pod.Status.Phase)
		} else {
			glog.Infof("%s/%s is running", ns, podName)
		}
	}

	// Wait for the endpoints to propagate.
	for start := time.Now(); time.Since(start) < endpointTimeout; time.Sleep(10 * time.Second) {
		_, err = http.Get(fmt.Sprintf("http://serve-hostnames.%s:9376", ns))
		if err == nil {
			break
		}
		glog.Infof("After %v while making a request got error %v", time.Since(start), err)
	}
	if err != nil {
		glog.Errorf("Failed to get a response from service: %v", err)
	}

	// Repeatedly make requests.
	for iteration := 0; iteration != *upTo; iteration++ {
		responseChan := make(chan string, queries)
		// Use a channel of size *maxPar to throttle the number
		// of in-flight requests to avoid overloading the service.
		inFlight := make(chan struct{}, *maxPar)
		start := time.Now()
		for q := 0; q < queries; q++ {
			go func(i int, query int) {
				inFlight <- struct{}{}
				t := time.Now()
				resp, err := http.Get(fmt.Sprintf("http://serve-hostnames.%s:9376", ns))
				glog.V(4).Infof("Call to serve-hostnames in namespace %s took %v", ns, time.Since(t))
				if err != nil {
					glog.Warningf("Call failed during iteration %d query %d : %v", i, query, err)
					// If the query failed return a string which starts with a character
					// that can't be part of a hostname.
					responseChan <- fmt.Sprintf("!failed in iteration %d to issue query %d: %v", i, query, err)
				} else {
					defer resp.Body.Close()
					hostname, err := ioutil.ReadAll(resp.Body)
					if err != nil {
						responseChan <- fmt.Sprintf("!failed in iteration %d to read body of response: %v", i, err)
					} else {
						responseChan <- string(hostname)
					}
				}
				<-inFlight
			}(iteration, q)
		}
		responses := make(map[string]int, *podsPerNode*len(nodes.Items))
		missing := 0
		for q := 0; q < queries; q++ {
			r := <-responseChan
			glog.V(4).Infof("Got response from %s", r)
			responses[r]++
			// If the returned hostname starts with '!' then it indicates
			// an error response.
			if len(r) > 0 && r[0] == '!' {
				glog.V(3).Infof("Got response %s", r)
				missing++
			}
		}
		if missing > 0 {
			glog.Warningf("Missing %d responses out of %d", missing, queries)
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
		glog.Infof("Iteration %d took %v for %d queries (%.2f QPS) with %d missing",
			iteration, time.Since(start), queries-missing, float64(queries-missing)/time.Since(start).Seconds(), missing)
	}
}
