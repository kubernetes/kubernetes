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
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/contrib/mesos/pkg/minion"
)

// NewMinion creates a new hyperkube Server object that includes the
// description and flags.
func NewMinion() *Server {
	s := minion.NewMinionServer()
	hks := Server{
		SimpleUsage: hyperkube.CommandMinion,
		Long:        `Implements a Kubernetes minion. This will launch the proxy and executor.`,
		Run: func(hks *Server, args []string) error {
			return s.Run(hks, args)
		},
	}
	s.AddMinionFlags(hks.Flags())
	s.AddExecutorFlags(hks.Flags())

	return &hks
}
