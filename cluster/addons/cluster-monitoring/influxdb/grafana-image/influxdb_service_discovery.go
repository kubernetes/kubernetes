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
	"fmt"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

func main() {
	glog.Info("influxdb service discovery")

	c, err := client.NewInCluster()
	if err != nil {
		glog.Fatalf("Failed to make client: %v", err)
	}

	glog.Info("Looking up monitoring-influxdb service...")
	var influxdb *api.Service
	// Wait a bit longer for influxdb loadBalancer to be created
	for t := time.Now(); time.Since(t) < 5*time.Minute; time.Sleep(10 * time.Second) {
		influxdb, err = c.Services(api.NamespaceSystem).Get("monitoring-influxdb")
		if err == nil && len(influxdb.Status.LoadBalancer.Ingress) > 0 {
			break
		} else if err == nil {
			glog.Info("Service is up. Waiting for load balancer to be created...")
		} else {
			glog.Info("Waiting for service to come up...")
		}
	}
	if influxdb == nil {
		glog.Error("Failed to find the monitoring-influxdb service.")
		return
	}

	var serviceIP string
	if len(influxdb.Status.LoadBalancer.Ingress) > 0 {
		ingress := influxdb.Status.LoadBalancer.Ingress[0]
		serviceIP = ingress.IP
		if serviceIP == "" {
			serviceIP = ingress.Hostname
		}
		glog.Infof("Found monitoring-influxdb service with external IP: '%s'", serviceIP)
	} else {
		glog.Warning("Service monitoring-influxdb exists but there's no load balancer.")
		return
	}

	var servicePort = 8086
	for _, port := range influxdb.Spec.Ports {
		if port.Name == "api" {
			servicePort = port.Port
			glog.Infof("Found API port for monitoring-influxdb service: %d", servicePort)
			break
		}
	}

	fmt.Printf("http://%s:%s", serviceIP, strconv.Itoa(servicePort))
}
