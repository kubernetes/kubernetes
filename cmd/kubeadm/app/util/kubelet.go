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

package util

import (
	"k8s.io/kubernetes/pkg/util/initsystem"
	"fmt"
)

func TryStartKubelet() {
	// If we notice that the kubelet service is inactive, try to start it
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		fmt.Println("[kubelet] No supported init system detected, won't ensure kubelet is running.")
	} else if initSystem.ServiceExists("kubelet") && !initSystem.ServiceIsActive("kubelet") {

		fmt.Println("[kubelet] Starting the kubelet service")
		if err := initSystem.ServiceStart("kubelet"); err != nil {
			fmt.Printf("[kubelet] WARNING: Unable to start the kubelet service: [%v]\n", err)
			fmt.Println("[kubelet] WARNING: Please ensure kubelet is running manually.")
		}
	}
}
