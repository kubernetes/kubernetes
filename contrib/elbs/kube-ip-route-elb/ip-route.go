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
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
)

func syncIPRoute(configs []*ELBConfig) {
	glog.Infof("Syncing ip routes (%d configs)...", len(configs))
	e := exec.New()

	for _, config := range configs {
		if len(config.Weights) == 0 {
			continue
		}
		args := make([]string, 0)
		args = append(args, "route", "replace", config.Target.Spec.ClusterIP+"/32", "table", "kube-svc")
		for _, hw := range config.Weights {
			args = append(args, "nexthop", "via", hw.Host.String(), "weight", fmt.Sprintf("%d", hw.Weight))
		}
		output, err := e.Command("ip", args...).CombinedOutput()
		if err != nil {
			glog.Errorf("ip %v: %v: %s", args, err, string(output))
		}
	}

	// TODO clean unused rules
}
