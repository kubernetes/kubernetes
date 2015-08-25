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

package main

import (
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

var (
	namespace = flag.String("namespace", api.NamespaceDefault, "The namespace containing Elasticsearch pods")
	selector  = flag.String("selector", "", "Selector (label query) for selecting Elasticsearch pods")
)

func main() {
	flag.Parse()
	glog.Info("Elasticsearch discovery")

	glog.Infof("Namespace: %q", *namespace)
	glog.Infof("selector: %q", *selector)

	c, err := client.NewInCluster()
	if err != nil {
		glog.Fatalf("Failed to make client: %v", err)
	}

	l, err := labels.Parse(*selector)
	if err != nil {
		glog.Fatalf("Failed to parse selector %q: %v", *selector, err)
	}
	pods, err := c.Pods(*namespace).List(l, fields.Everything())
	if err != nil {
		glog.Fatalf("Failed to list pods: %v", err)
	}

	glog.Infof("Elasticsearch pods in namespace %s with selector %q", *namespace, *selector)
	podIPs := []string{}
	for i := range pods.Items {
		p := &pods.Items[i]
		for attempt := 0; attempt < 10; attempt++ {
			glog.Infof("%d: %s PodIP: %s", i, p.Name, p.Status.PodIP)
			if p.Status.PodIP != "" {
				podIPs = append(podIPs, fmt.Sprintf(`"%s"`, p.Status.PodIP))
				break
			}
			time.Sleep(1 * time.Second)
			p, err = c.Pods(*namespace).Get(p.Name)
			if err != nil {
				glog.Warningf("Failed to get pod %s: %v", p.Name, err)
			}
		}
		if p.Status.PodIP == "" {
			glog.Warningf("Failed to obtain PodIP for %s", p.Name)
		}
	}
	fmt.Printf("discovery.zen.ping.unicast.hosts: [%s]\n", strings.Join(podIPs, ", "))
}
