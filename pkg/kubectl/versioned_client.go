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
	clientappsv1beta1 "k8s.io/client-go/kubernetes/typed/apps/v1beta1"
	clientextensionsv1beta1 "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	internalclientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

// TODO: get rid of this and plumb the caller correctly
func versionedExtensionsClientV1beta1(internalClient internalclientset.Interface) clientextensionsv1beta1.ExtensionsV1beta1Interface {
	if internalClient == nil {
		return &clientextensionsv1beta1.ExtensionsV1beta1Client{}
	}
	return clientextensionsv1beta1.New(internalClient.Extensions().RESTClient())
}

// TODO: get rid of this and plumb the caller correctly
func versionedAppsClientV1beta1(internalClient internalclientset.Interface) clientappsv1beta1.AppsV1beta1Interface {
	if internalClient == nil {
		return &clientappsv1beta1.AppsV1beta1Client{}
	}
	return clientappsv1beta1.New(internalClient.Apps().RESTClient())
}
