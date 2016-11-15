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
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

// NewKubeAPIServer creates a new hyperkube Server object that includes the
// description and flags.
func NewKubeAPIServer() *Server {
	s := options.NewServerRunOptions()

	hks := Server{
		SimpleUsage: "apiserver",
		Long:        "The main API entrypoint and interface to the storage system.  The API server is also the focal point for all authorization decisions.",
		Run: func(_ *Server, args []string) error {
			return app.Run(s)
		},
	}
	s.AddFlags(hks.Flags())
	return &hks
}
