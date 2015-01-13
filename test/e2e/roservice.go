/*
Copyright 2014 Google Inc. All rights reserved.

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

package e2e

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/golang/glog"
)

func TestKubernetesROService(c *client.Client) bool {
	svc := api.ServiceList{}
	err := c.Get().
		Namespace("default").
		AbsPath("/api/v1beta1/proxy/services/kubernetes-ro/api/v1beta1/services").
		Do().
		Into(&svc)
	if err != nil {
		glog.Errorf("unexpected error listing services using ro service: %v", err)
		return false
	}
	var foundRW, foundRO bool
	for i := range svc.Items {
		if svc.Items[i].Name == "kubernetes" {
			foundRW = true
		}
		if svc.Items[i].Name == "kubernetes-ro" {
			foundRO = true
		}
	}
	if !foundRW {
		glog.Error("no RW service found")
	}
	if !foundRO {
		glog.Error("no RO service found")
	}
	if !foundRW || !foundRO {
		return false
	}
	return true
}
