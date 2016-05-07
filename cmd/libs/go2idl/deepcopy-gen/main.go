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

// deepcopy-gen is a tool for auto-generating DeepCopy functions.
//
// Structs in the input directories with the below line in their comments
// will be ignored during generation.
// // +gencopy=false
package main

import (
	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/deepcopy-gen/generators"

	"github.com/golang/glog"
)

func main() {
	arguments := args.Default()

	// Override defaults. These are Kubernetes specific input locations.
	arguments.InputDirs = []string{
		"k8s.io/kubernetes/pkg/api",
		"k8s.io/kubernetes/pkg/api/v1",
		"k8s.io/kubernetes/pkg/apis/authorization",
		"k8s.io/kubernetes/pkg/apis/authorization/v1beta1",
		"k8s.io/kubernetes/pkg/apis/autoscaling",
		"k8s.io/kubernetes/pkg/apis/autoscaling/v1",
		"k8s.io/kubernetes/pkg/apis/batch",
		"k8s.io/kubernetes/pkg/apis/batch/v1",
		"k8s.io/kubernetes/pkg/apis/apps",
		"k8s.io/kubernetes/pkg/apis/apps/v1alpha1",
		"k8s.io/kubernetes/pkg/apis/componentconfig",
		"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1",
		"k8s.io/kubernetes/pkg/apis/policy",
		"k8s.io/kubernetes/pkg/apis/policy/v1alpha1",
		"k8s.io/kubernetes/pkg/apis/extensions",
		"k8s.io/kubernetes/pkg/apis/extensions/v1beta1",
		"k8s.io/kubernetes/pkg/apis/metrics",
		"k8s.io/kubernetes/pkg/apis/metrics/v1alpha1",
		"k8s.io/kubernetes/federation/apis/federation",
		"k8s.io/kubernetes/federation/apis/federation/v1alpha1",
	}

	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
	glog.Info("Completed successfully.")
}
