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
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/service"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
)

// NewHyperkubeServer creates a new hyperkube Server object that includes the
// description and flags.
func NewKubeletExecutor() *Server {
	s := service.NewKubeletExecutorServer()
	hks := Server{
		SimpleUsage: hyperkube.CommandExecutor,
		Long: `The kubelet-executor binary is responsible for maintaining a set of containers
on a particular node. It syncs data from a specialized Mesos source that tracks
task launches and kills. It then queries Docker to see what is currently
running.  It synchronizes the configuration data, with the running set of
containers by starting or stopping Docker containers.`,
		Run: func(hks *Server, args []string) error {
			return s.Run(hks, args)
		},
	}
	s.AddFlags(hks.Flags())
	return &hks
}
