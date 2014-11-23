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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// MasterFlagsNamespacer can return a MasterFlagsInterface for the given namespace.
type MasterFlagsNamespacer interface {
	MasterFlags(namespace string) MasterFlagsInterface
}

// MasterFlagsInterface has methods to work with MasterFlags resources
type MasterFlagsInterface interface {
	Create(masterFlags *api.MasterFlags) (*api.MasterFlags, error)
	List(label, field labels.Selector) (*api.MasterFlagsList, error)
	Get(id string) (*api.MasterFlags, error)
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// masterFlags implements MasterFlags interface
type masterFlags struct {
	client    *Client
	namespace string
}

// newMasterFlags returns a new masterFlags object.
func newMasterFlags(c *Client, ns string) *masterFlags {
	return &masterFlags{
		client:    c,
		namespace: ns,
	}
}

// Create makes a new masterFlags. Returns the copy of the masterFlags the server returns,
// or an error. The namespace to create the masterFlags within is deduced from the
// masterFlags; it must either match this masterFlags client's namespace, or this masterFlags
// client must have been created with the "" namespace.
func (f *masterFlags) Create(masterFlags *api.MasterFlags) (*api.MasterFlags, error) {
	if f.namespace != "" && masterFlags.Namespace != f.namespace {
		return nil, fmt.Errorf("can't create a masterFlags with namespace '%v' in namespace '%v'", masterFlags.Namespace, f.namespace)
	}
	result := &api.MasterFlags{}
	err := f.client.Post().
		Path("masterFlags").
		Namespace(masterFlags.Namespace).
		Body(masterFlags).
		Do().
		Into(result)
	return result, err
}

// List returns a list of masterFlags matching the selectors.
func (f *masterFlags) List(label, field labels.Selector) (*api.MasterFlagsList, error) {
	result := &api.MasterFlagsList{}
	err := f.client.Get().
		Path("masterFlags").
		Namespace(f.namespace).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Do().
		Into(result)
	return result, err
}

// Get returns the given masterFlags, or an error.
func (f *masterFlags) Get(id string) (*api.MasterFlags, error) {
	result := &api.MasterFlags{}
	err := f.client.Get().
		Path("masterFlags").
		Path(id).
		Namespace(f.namespace).
		Do().
		Into(result)
	return result, err
}

// Watch starts watching for masterFlags matching the given selectors.
func (f *masterFlags) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return f.client.Get().
		Path("watch").
		Path("masterFlags").
		Param("resourceVersion", resourceVersion).
		Namespace(f.namespace).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Watch()
}
