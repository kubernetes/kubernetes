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

package rest

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// ResourceInterface is the interface for the parts of the master that know how to add/remove
// third party resources.  Extracted into an interface for injection for testing.
type ResourceInterface interface {
	// Remove a third party resource based on the RESTful path for that resource, the path is <api-group-path>/<resource-plural-name>
	RemoveThirdPartyResource(path string) error
	// Install a third party resource described by 'rsrc'
	InstallThirdPartyResource(rsrc *extensions.ThirdPartyResource) error
	// Is a particular third party resource currently installed?
	HasThirdPartyResource(rsrc *extensions.ThirdPartyResource) (bool, error)
	// List all currently installed third party resources, the returned
	// names are of the form <api-group-path>/<resource-plural-name>
	ListThirdPartyResources() []string
}

const thirdpartyprefix = "/apis"

// ThirdPartyController is a control loop that knows how to synchronize ThirdPartyResource objects with
// RESTful resources which are present in the API server.
type ThirdPartyController struct {
	master ResourceInterface
	client extensionsclient.ThirdPartyResourcesGetter
}

// SyncOneResource synchronizes a single resource with RESTful resources on the master
func (t *ThirdPartyController) SyncOneResource(rsrc *extensions.ThirdPartyResource) error {
	// TODO: we also need to test if the existing installed resource matches the resource we are sync-ing.
	// Currently, if there is an older, incompatible resource installed, we won't remove it.  We should detect
	// older, incompatible resources and remove them before testing if the resource exists.
	hasResource, err := t.master.HasThirdPartyResource(rsrc)
	if err != nil {
		return err
	}
	if !hasResource {
		return t.master.InstallThirdPartyResource(rsrc)
	}
	return nil
}

// Synchronize all resources with RESTful resources on the master
func (t *ThirdPartyController) SyncResources() error {
	list, err := t.client.ThirdPartyResources().List(api.ListOptions{})
	if err != nil {
		return err
	}
	return t.syncResourceList(list)
}

func (t *ThirdPartyController) syncResourceList(list runtime.Object) error {
	existing := sets.String{}
	switch list := list.(type) {
	case *extensions.ThirdPartyResourceList:
		// Loop across all schema objects for third party resources
		for ix := range list.Items {
			item := &list.Items[ix]
			// extract the api group and resource kind from the schema
			_, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(item)
			if err != nil {
				return err
			}
			// place it in the set of resources that we expect, so that we don't delete it in the delete pass
			existing.Insert(MakeThirdPartyPath(group))
			// ensure a RESTful resource for this schema exists on the master
			if err := t.SyncOneResource(item); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("expected a *ThirdPartyResourceList, got %#v", list)
	}
	// deletion phase, get all installed RESTful resources
	installed := t.master.ListThirdPartyResources()
	for _, installedAPI := range installed {
		found := false
		// search across the expected restful resources to see if this resource belongs to one of the expected ones
		for _, apiPath := range existing.List() {
			if installedAPI == apiPath || strings.HasPrefix(installedAPI, apiPath+"/") {
				found = true
				break
			}
		}
		// not expected, delete the resource
		if !found {
			if err := t.master.RemoveThirdPartyResource(installedAPI); err != nil {
				return err
			}
		}
	}

	return nil
}

func MakeThirdPartyPath(group string) string {
	if len(group) == 0 {
		return thirdpartyprefix
	}
	return thirdpartyprefix + "/" + group
}

func GetThirdPartyGroupName(path string) string {
	return strings.TrimPrefix(strings.TrimPrefix(path, thirdpartyprefix), "/")
}
