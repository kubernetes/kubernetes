/*
Copyright 2017 The Kubernetes Authors.

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
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog"
	utilnet "k8s.io/utils/net"
)

func buildConfigFromEnvs(masterURL, kubeconfigPath string) (*restclient.Config, error) {
	if kubeconfigPath == "" && masterURL == "" {
		kubeconfig, err := restclient.InClusterConfig()
		if err != nil {
			return nil, err
		}

		return kubeconfig, nil
	}

	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: kubeconfigPath},
		&clientcmd.ConfigOverrides{ClusterInfo: clientapi.Cluster{Server: masterURL}}).ClientConfig()
}

func flattenSubsets(subsets []corev1.EndpointSubset) []string {
	ips := []string{}
	for _, ss := range subsets {
		for _, addr := range ss.Addresses {
			if utilnet.IsIPv6String(addr.IP) {
				ips = append(ips, fmt.Sprintf(`"[%s]"`, addr.IP))
			} else {
				ips = append(ips, fmt.Sprintf(`"%s"`, addr.IP))
			}
		}
	}
	return ips
}

func getAdvertiseAddress() (string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}

	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			return ipnet.IP.String(), nil
		}
	}

	return "", fmt.Errorf("no non-loopback address is available")
}

func main() {
	flag.Parse()

	klog.Info("Kubernetes Elasticsearch logging discovery")

	advertiseAddress, err := getAdvertiseAddress()
	if err != nil {
		klog.Fatalf("Failed to get valid advertise address: %v", err)
	}
	fmt.Printf("network.host: \"%s\"\n\n", advertiseAddress)

	cc, err := buildConfigFromEnvs(os.Getenv("APISERVER_HOST"), os.Getenv("KUBE_CONFIG_FILE"))
	if err != nil {
		klog.Fatalf("Failed to make client: %v", err)
	}
	client, err := clientset.NewForConfig(cc)

	if err != nil {
		klog.Fatalf("Failed to make client: %v", err)
	}
	namespace := metav1.NamespaceSystem
	envNamespace := os.Getenv("NAMESPACE")
	if envNamespace != "" {
		if _, err := client.CoreV1().Namespaces().Get(envNamespace, metav1.GetOptions{}); err != nil {
			klog.Fatalf("%s namespace doesn't exist: %v", envNamespace, err)
		}
		namespace = envNamespace
	}

	var elasticsearch *corev1.Service
	serviceName := os.Getenv("ELASTICSEARCH_SERVICE_NAME")
	if serviceName == "" {
		serviceName = "elasticsearch-logging"
	}

	// Look for endpoints associated with the Elasticsearch logging service.
	// First wait for the service to become available.
	for t := time.Now(); time.Since(t) < 5*time.Minute; time.Sleep(10 * time.Second) {
		elasticsearch, err = client.CoreV1().Services(namespace).Get(serviceName, metav1.GetOptions{})
		if err == nil {
			break
		}
	}
	// If we did not find an elasticsearch logging service then log a warning
	// and return without adding any unicast hosts.
	if elasticsearch == nil {
		klog.Warningf("Failed to find the elasticsearch-logging service: %v", err)
		return
	}

	var endpoints *corev1.Endpoints
	addrs := []string{}
	// Wait for some endpoints.
	count, _ := strconv.Atoi(os.Getenv("MINIMUM_MASTER_NODES"))
	for t := time.Now(); time.Since(t) < 5*time.Minute; time.Sleep(10 * time.Second) {
		endpoints, err = client.CoreV1().Endpoints(namespace).Get(serviceName, metav1.GetOptions{})
		if err != nil {
			continue
		}
		addrs = flattenSubsets(endpoints.Subsets)
		klog.Infof("Found %s", addrs)
		if len(addrs) > 0 && len(addrs) >= count {
			break
		}
	}
	// If there was an error finding endpoints then log a warning and quit.
	if err != nil {
		klog.Warningf("Error finding endpoints: %v", err)
		return
	}

	klog.Infof("Endpoints = %s", addrs)
	fmt.Printf("discovery.seed_hosts: [%s]\n", strings.Join(addrs, ", "))
	fmt.Printf("cluster.initial_master_nodes: [%s]\n", strings.Join(addrs, ", "))
}
