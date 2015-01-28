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

package client

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// ResourceQuotaUsagesNamespacer has methods to work with ResourceQuotaUsage resources in a namespace
type ResourceQuotaUsagesNamespacer interface {
	ResourceQuotaUsages(namespace string) ResourceQuotaUsageInterface
}

// ResourceQuotaUsageInterface has methods to work with ResourceQuotaUsage resources.
type ResourceQuotaUsageInterface interface {
	Create(resourceQuotaUsage *api.ResourceQuotaUsage) error
}

// resourceQuotaUsages implements ResourceQuotaUsagesNamespacer interface
type resourceQuotaUsages struct {
	r  *Client
	ns string
}

// newResourceQuotaUsages returns a resourceQuotaUsages
func newResourceQuotaUsages(c *Client, namespace string) *resourceQuotaUsages {
	return &resourceQuotaUsages{
		r:  c,
		ns: namespace,
	}
}

// Create takes the representation of a resourceQuotaUsage.  Returns an error if the usage was not applied
func (c *resourceQuotaUsages) Create(resourceQuotaUsage *api.ResourceQuotaUsage) (err error) {
	if len(resourceQuotaUsage.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", resourceQuotaUsage)
		return
	}
	err = c.r.Post().Namespace(c.ns).Resource("resourceQuotaUsages").Body(resourceQuotaUsage).Do().Error()
	return
}
