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
	"os"

	"k8s.io/kube-aggregator/pkg/cmd/server"
)

// NewKubeAggregator creates a new hyperkube Server object that includes the
// description and flags.
func NewKubeAggregator() *Server {
	o := server.NewDefaultOptions(os.Stdout, os.Stderr)

	hks := Server{
		name:            "aggregator",
		AlternativeName: "kube-aggregator",
		SimpleUsage:     "aggregator",
		Long:            "Aggregator for Kubernetes-style API servers: dynamic registration, discovery summarization, secure proxy.",
		Run: func(_ *Server, args []string, stopCh <-chan struct{}) error {
			if err := o.Complete(); err != nil {
				return err
			}
			if err := o.Validate(args); err != nil {
				return err
			}
			if err := o.RunAggregator(stopCh); err != nil {
				return err
			}
			return nil
		},
		RespectsStopCh: true,
	}

	o.AddFlags(hks.Flags())
	return &hks
}
