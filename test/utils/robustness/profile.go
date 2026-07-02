/*
Copyright The Kubernetes Authors.

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

package robustness

import "fmt"

// ResourceRef identifies an API resource — and optionally a single instance of it
// — the way the apiserver's RequestInfo reports it: by API group and plural
// resource name. Namespace and Name narrow the reference to one object; left
// empty they match any instance.
type ResourceRef struct {
	Group     string // API group, e.g. "apps" ("" for the core group)
	Resource  string // plural resource name, e.g. "daemonsets"
	Namespace string // optional; "" matches any namespace
	Name      string // optional; "" matches any instance
}

// ChildResource describes the dependent resource a controller manages on behalf
// of its root object (e.g. the Pods a DaemonSet controller creates).
type ChildResource struct {
	Group    string // API group of the child ("" for the core group)
	Resource string // plural resource name, e.g. "pods"

	// CacheName is the name under which the child's informer cache is wrapped
	// (see RobustnessTestFixture.WrapIndexer and the informer helpers). Leave it
	// empty if the cache is not wrapped; cache-lag scenarios are then skipped.
	CacheName string

	// CreatedByController is true when the controller POSTs child objects itself
	// through the wrapped client, enabling write-failure faults on the
	// child-create path.
	CreatedByController bool
}

// ControllerProfile declares what the controller-under-test does through the
// wrapped client. Scenarios consult the profile to inject faults only at sites
// the controller actually exercises; combined with the unmatched-fault guard
// (AssertAllFaultsMatched) this keeps the chaos matrix honest for any
// controller: a fault is either injected where it can really land, or not
// registered at all — never a silent no-op.
//
// The declarations are promises: if the profile says WritesRootStatus, the
// matrix will require the corresponding conflict fault to actually fire.
type ControllerProfile struct {
	// Name identifies the controller in logs.
	Name string

	// Root is the object whose reconciliation is under test.
	Root ResourceRef

	// WritesRoot is true when the controller PUTs the root object itself.
	WritesRoot bool

	// WritesRootStatus is true when the controller PUTs the root object's
	// /status subresource.
	WritesRootStatus bool

	// Child, if non-nil, is the dependent resource the controller manages.
	Child *ChildResource

	// UsesExpectations is true when the controller tracks in-flight creates and
	// deletes with ControllerExpectations, enabling expectations-timeout faults
	// on the wrapped expectations clock.
	UsesExpectations bool
}

func (p ControllerProfile) validate() error {
	if p.Root.Resource == "" {
		return fmt.Errorf("ControllerProfile.Root.Resource must be set to the plural resource name (e.g. %q)", "daemonsets")
	}
	if p.Child != nil && p.Child.Resource == "" {
		return fmt.Errorf("ControllerProfile.Child.Resource must be set to the plural resource name (e.g. %q)", "pods")
	}
	return nil
}

// HasChildCache reports whether the controller has a child resource with a
// wrapped informer cache, i.e. whether cache faults can target it.
func (p ControllerProfile) HasChildCache() bool {
	return p.Child != nil && p.Child.CacheName != ""
}

// CreatesChildren reports whether the controller creates child objects itself.
func (p ControllerProfile) CreatesChildren() bool {
	return p.Child != nil && p.Child.CreatedByController
}

// RootWriteMatch selects the controller's PUTs of the root object itself
// (Subresource is exact-matched, so this never bleeds into /status).
func (p ControllerProfile) RootWriteMatch() ClientMatch {
	return ClientMatch{
		Verb:      "PUT",
		Group:     p.Root.Group,
		Resource:  p.Root.Resource,
		Namespace: p.Root.Namespace,
		Name:      p.Root.Name,
	}
}

// RootStatusWriteMatch selects the controller's PUTs of the root object's
// /status subresource.
func (p ControllerProfile) RootStatusWriteMatch() ClientMatch {
	m := p.RootWriteMatch()
	m.Subresource = "status"
	return m
}

// ChildCreateMatch selects the controller's POSTs of child objects. Child must
// be set (see CreatesChildren).
func (p ControllerProfile) ChildCreateMatch() ClientMatch {
	return ClientMatch{Verb: "POST", Group: p.Child.Group, Resource: p.Child.Resource}
}

// ChildCacheMatch selects lookups in the child's wrapped informer cache. Child
// must be set with a CacheName (see HasChildCache).
func (p ControllerProfile) ChildCacheMatch() CacheMatch {
	return CacheMatch{Cache: p.Child.CacheName}
}
