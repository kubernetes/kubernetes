/*
Copyright 2016 The Kubernetes Authors.

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
	"flag"

	"github.com/spf13/pflag"
	"k8s.io/code-generator/cmd/informer-gen/generators"
	"k8s.io/code-generator/pkg/util"
	"k8s.io/klog/v2"

	generatorargs "k8s.io/code-generator/cmd/informer-gen/args"
)

func main() {
	klog.InitFlags(nil)
	genericArgs, customArgs := generatorargs.NewDefaults()

	// Override defaults.
	// TODO: move out of informer-gen
	genericArgs.GoHeaderFilePath = util.BoilerplatePath()
	genericArgs.OutputPackagePath = "k8s.io/kubernetes/pkg/client/informers/informers_generated"
	customArgs.VersionedClientSetPackage = "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	customArgs.InternalClientSetPackage = "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	customArgs.ListersPackage = "k8s.io/kubernetes/pkg/client/listers"

	genericArgs.AddFlags(pflag.CommandLine)
	customArgs.AddFlags(pflag.CommandLine)
	flag.Set("logtostderr", "true")
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	if err := generatorargs.Validate(genericArgs); err != nil {
		klog.Fatalf("Error: %v", err)
	}

	// Run it.
	if err := genericArgs.Execute(
		generators.NameSystems(util.PluralExceptionListToMapOrDie(customArgs.PluralExceptions)),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	klog.V(2).Info("Completed successfully.")
}
