/*
Copyright 2020 The Kubernetes Authors.

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

// The addon manager ensures that a given set of addons is installed in the cluster.
package main

import (
	"math/rand"
	"os"
	"time"

	"k8s.io/component-base/logs"
	"k8s.io/kubernetes/cmd/kube-addon-manager/app"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	cmd := app.NewAddonManagerCommand()

	logs.InitLogs()
	defer logs.FlushLogs()

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
