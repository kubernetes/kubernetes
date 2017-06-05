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

package kubeconfig

/*

	PHASE: KUBECONFIG

	INPUTS:
		From MasterConfiguration
			The Master API Server endpoint (AdvertiseAddress + BindPort) is required so the KubeConfig file knows where to find the master
			The KubernetesDir path is required for knowing where to put the KubeConfig files
			The PKIPath is required for knowing where all certificates should be stored

	OUTPUTS:
		Files to KubernetesDir (default /etc/kubernetes):
		 - admin.conf
		 - kubelet.conf
		 - scheduler.conf
		 - controller-manager.conf
*/
