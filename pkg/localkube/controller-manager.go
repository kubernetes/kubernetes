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
	controllermanager "k8s.io/kubernetes/cmd/kube-controller-manager/app"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
)

func (lk LocalkubeServer) NewControllerManagerServer() Server {
	return NewSimpleServer("controller-manager", serverInterval, StartControllerManagerServer(lk))
}

func StartControllerManagerServer(lk LocalkubeServer) func() error {
	config := options.NewCMServer()

	config.Master = lk.GetAPIServerInsecureURL()

	// defaults from command
	config.DeletingPodsQps = 0.1
	config.DeletingPodsBurst = 10
	config.NodeEvictionRate = 0.1

	config.EnableProfiling = true
	config.VolumeConfiguration.EnableHostPathProvisioning = true
	config.VolumeConfiguration.EnableDynamicProvisioning = true
	config.ServiceAccountKeyFile = lk.GetPrivateKeyCertPath()
	config.RootCAFile = lk.GetCAPublicKeyCertPath()

	lk.SetExtraConfigForComponent("controller-manager", &config)

	return func() error {
		return controllermanager.Run(config)
	}
}
