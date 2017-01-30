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
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/examples/apiserver"
	"k8s.io/kubernetes/pkg/util/flag"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

func main() {
	serverRunOptions := apiserver.NewServerRunOptions()

	// Parse command line flags.
	serverRunOptions.GenericServerRunOptions.AddUniversalFlags(pflag.CommandLine)
	serverRunOptions.Etcd.AddFlags(pflag.CommandLine)
	serverRunOptions.SecureServing.AddFlags(pflag.CommandLine)
	serverRunOptions.SecureServing.AddDeprecatedFlags(pflag.CommandLine)
	serverRunOptions.InsecureServing.AddFlags(pflag.CommandLine)
	serverRunOptions.InsecureServing.AddDeprecatedFlags(pflag.CommandLine)
	flag.InitFlags()

	if err := serverRunOptions.Run(wait.NeverStop); err != nil {
		glog.Fatalf("Error in bringing up the server: %v", err)
	}
}
