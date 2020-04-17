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
	"os"

	"k8s.io/klog"

	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/component-base/logs"
	"k8s.io/kube-aggregator/pkg/cmd/server"

	// force compilation of packages we'll later rely upon
	_ "k8s.io/kube-aggregator/pkg/apis/apiregistration/install"
	_ "k8s.io/kube-aggregator/pkg/apis/apiregistration/validation"
	_ "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	_ "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1"
	_ "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1beta1"
)

func main() {
	logs.InitLogs()
	defer logs.FlushLogs()

	stopCh := genericapiserver.SetupSignalHandler()
	options := server.NewDefaultOptions(os.Stdout, os.Stderr)
	cmd := server.NewCommandStartAggregator(options, stopCh)
	cmd.Flags().AddGoFlagSet(flag.CommandLine)
	if err := cmd.Execute(); err != nil {
		klog.Fatal(err)
	}
}
