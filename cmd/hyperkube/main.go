/*
Copyright 2014 The Kubernetes Authors.

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

// A binary that can morph into all of the other kubernetes binaries. You can
// also soft-link to it busybox style.
//
// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/cmd/km/km.go
package main

import (
	"os"
)

func main() {
	hk := HyperKube{
		Name: "hyperkube",
		Long: "This is an all-in-one binary that can run any of the various Kubernetes servers.",
	}

	hk.AddServer(NewKubectlServer())
	hk.AddServer(NewKubeAPIServer())
	hk.AddServer(NewKubeControllerManager())
	hk.AddServer(NewScheduler())
	hk.AddServer(NewKubelet())
	hk.AddServer(NewKubeProxy())

	//Federation servers
	hk.AddServer(NewFederationAPIServer())
	hk.AddServer(NewFederationCMServer())

	hk.RunToExit(os.Args)
}
