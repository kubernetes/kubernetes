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

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

func flattenSubsets(subsets []api.EndpointSubset) []string {
	ips := []string{}
	for _, ss := range subsets {
		for _, addr := range ss.Addresses {
			ips = append(ips, fmt.Sprintf(`"%s"`, addr.IP))
		}
	}
	return ips
}

func main() {
	flag.Parse()
	glog.Info("Kubernetes Elasticsearch logging discovery")

	c, err := client.NewInCluster()
	if err != nil {
		glog.Fatalf("Failed to make client: %v", err)
	}
	namespace := api.NamespaceSystem
	envNamespace := os.Getenv("NAMESPACE")
	if envNamespace != "" {
		if _, err := c.Namespaces().Get(envNamespace); err != nil {
			glog.Fatalf("%s namespace doesn't exist: %v", envNamespace, err)
		}
		namespace = envNamespace
	}

	var elasticsearch *api.Service
	// Look for endpoints associated with the Elasticsearch loggging service.
	// First wait for the service to become available.
	for t := time.Now(); time.Since(t) < 5*time.Minute; time.Sleep(10 * time.Second) {
		elasticsearch, err = c.Services(namespace).Get("elasticsearch-logging")
		if err == nil {
			break
		}
	}
	// If we did not find an elasticsearch logging service then log a warning
	// and return without adding any unicast hosts.
	if elasticsearch == nil {
		glog.Warningf("Failed to find the elasticsearch-logging service: %v", err)
		return
	}

	var endpoints *api.Endpoints
	addrs := []string{}
	// Wait for some endpoints.
	count := 0
	for t := time.Now(); time.Since(t) < 5*time.Minute; time.Sleep(10 * time.Second) {
		endpoints, err = c.Endpoints(namespace).Get("elasticsearch-logging")
		if err != nil {
			continue
		}
		addrs = flattenSubsets(endpoints.Subsets)
		glog.Infof("Found %s", addrs)
		if len(addrs) > 0 && len(addrs) == count {
			break
		}
		count = len(addrs)
	}
	// If there was an error finding endpoints then log a warning and quit.
	if err != nil {
		glog.Warningf("Error finding endpoints: %v", err)
		return
	}

	glog.Infof("Endpoints = %s", addrs)
	fmt.Printf("discovery.zen.ping.unicast.hosts: [%s]\n", strings.Join(addrs, ", "))
}
