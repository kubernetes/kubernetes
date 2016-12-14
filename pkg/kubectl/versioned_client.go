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

package kubectl

import (
	externalclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	extensions "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	internalclientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

func versionedClientsetForDeployment(internalClient internalclientset.Interface) externalclientset.Interface {
	if internalClient == nil {
		return &externalclientset.Clientset{}
	}
	return &externalclientset.Clientset{
		CoreV1Client:            core.New(internalClient.Core().RESTClient()),
		ExtensionsV1beta1Client: extensions.New(internalClient.Extensions().RESTClient()),
	}
}
