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

package openstack

import (
	"errors"

	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

func init() {
	if err := rest.RegisterAuthProviderPlugin("openstack", newOpenstackAuthProvider); err != nil {
		klog.Fatalf("Failed to register openstack auth plugin: %s", err)
	}
}

func newOpenstackAuthProvider(_ string, _ map[string]string, _ rest.AuthProviderConfigPersister) (rest.AuthProvider, error) {
	return nil, errors.New(`The openstack auth plugin has been removed.
Please use the "client-keystone-auth" kubectl/client-go credential plugin instead.
See https://github.com/kubernetes/cloud-provider-openstack/blob/master/docs/using-client-keystone-auth.md for further details`)
}
