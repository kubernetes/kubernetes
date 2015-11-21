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
	"os"

	"k8s.io/kubernetes/cmd/kubectl/app"
)

func NewKubectlServer() *Server {
	return &Server{
		name:        "kubectl",
		SimpleUsage: "Kubernetes command line client",
		Long:        "Kubernetes command line client",
		Run: func(s *Server, args []string) error {
			os.Args = os.Args[1:]
			if err := app.Run(); err != nil {
				os.Exit(1)
			}
			os.Exit(0)
			return nil
		},
	}
}
