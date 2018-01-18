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
	"k8s.io/kubernetes/cmd/cloud-controller-manager/app"
	"k8s.io/kubernetes/cmd/cloud-controller-manager/app/options"
)

// NewCloudControllerManager creates a new hyperkube Server object that includes the
// description and flags.
func NewCloudControllerManager() *Server {
	s := options.NewCloudControllerManagerServer()

	hks := Server{
		name:        "cloud-controller-manager",
		SimpleUsage: "cloud-controller-manager",
		Long:        "A server that acts as an external cloud provider.",
		Run: func(_ *Server, args []string, stopCh <-chan struct{}) error {
			return app.Run(s)
		},
	}
	s.AddFlags(hks.Flags())
	return &hks
}
