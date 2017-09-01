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
	"k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
)

// NewKubelet creates a new hyperkube Server object that includes the
// description and flags.
func NewKubelet() (*Server, error) {
	s, err := options.NewKubeletServer()
	if err != nil {
		return nil, err
	}

	hks := Server{
		name:        "kubelet",
		SimpleUsage: "kubelet",
		Long: `The kubelet binary is responsible for maintaining a set of containers on a
		particular node. It syncs data from a variety of sources including a
		Kubernetes API server, an etcd cluster, HTTP endpoint or local file. It then
		queries Docker to see what is currently running.  It synchronizes the
		configuration data, with the running set of containers by starting or stopping
		Docker containers.`,
		Run: func(_ *Server, _ []string, stopCh <-chan struct{}) error {
			return app.Run(s, nil)
		},
	}
	s.AddFlags(hks.Flags())
	return &hks, nil
}
