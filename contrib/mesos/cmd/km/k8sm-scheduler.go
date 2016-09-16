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

// clone of the upstream cmd/hypercube/k8sm-scheduler.go
package main

import (
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/service"
)

// NewScheduler creates a new hyperkube Server object that includes the
// description and flags.
func NewScheduler() *Server {
	s := service.NewSchedulerServer()

	hks := Server{
		SimpleUsage: hyperkube.CommandScheduler,
		Long: `Implements the Kubernetes-Mesos scheduler. This will launch Mesos tasks which
results in pods assigned to kubelets based on capacity and constraints.`,
		Run: func(hks *Server, args []string) error {
			return s.Run(hks, args)
		},
	}
	s.AddHyperkubeFlags(hks.Flags())
	return &hks
}
