/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package e2e_node

import (
	"flag"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/test/e2e/framework"
)

var kubeletAddress = flag.String("kubelet-address", "http://127.0.0.1:10255", "Host and port of the kubelet")
var apiServerAddress = flag.String("api-server-address", "http://127.0.0.1:8080", "Host and port of the api server")
var nodeName = flag.String("node-name", "", "Name of the node")
var buildServices = flag.Bool("build-services", true, "If true, build local executables")
var startServices = flag.Bool("start-services", true, "If true, start local node services")
var stopServices = flag.Bool("stop-services", true, "If true, stop local node services after running tests")

func NewDefaultFramework(baseName string) *framework.Framework {
	// Provides a client config for the framework to create a client.
	f := func() (*restclient.Config, error) {
		return &restclient.Config{Host: *apiServerAddress}, nil
	}
	return framework.NewFrameworkWithConfigGetter(baseName,
		framework.FrameworkOptions{
			ClientQPS:   100,
			ClientBurst: 100,
		}, nil, f)
}

func assignPodToNode(pod *api.Pod) {
	pod.Spec.NodeName = *nodeName
}
