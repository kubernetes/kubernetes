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
	"flag"

	"github.com/golang/glog"
	"k8s.io/kubernetes/cluster/addons/iso-client/isolator"
	"k8s.io/kubernetes/cluster/addons/iso-client/noop"
)

const (
	// kubelet eventDispatcher address
	remoteAddress = "localhost:5433"
	// iso-client own address
	localAddress = "localhost:5444"
	// name of isolator
	name = "noop"
)

func main() {
	// enable logging to STDERR
	flag.Set("logtostderr", "true")
	flag.Parse()
	glog.Info("Starting ...")

	// Create isolator
	iso, err := noop.New(name)
	if err != nil {
		glog.Fatalf("Cannot create isolator: %q", err)
	}

	// Start grpc server to handle isolation events
	err = isolator.StartIsolatorServer(iso, localAddress, remoteAddress)
	if err != nil {
		glog.Fatalf("Couldn't initialize isolator: %v", err)
	}
}
