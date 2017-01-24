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

package localkube

import (
	scheduler "k8s.io/kubernetes/plugin/cmd/kube-scheduler/app"
	"k8s.io/kubernetes/plugin/cmd/kube-scheduler/app/options"
)

func (lk LocalkubeServer) NewSchedulerServer() Server {
	return NewSimpleServer("scheduler", serverInterval, StartSchedulerServer(lk))
}

func StartSchedulerServer(lk LocalkubeServer) func() error {
	config := options.NewSchedulerServer()

	// master details
	config.Master = lk.GetAPIServerInsecureURL()

	// defaults from command
	config.EnableProfiling = true

	lk.SetExtraConfigForComponent("scheduler", &config)

	return func() error {
		return scheduler.Run(config)
	}
}
