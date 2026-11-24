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

import (
	"fmt"
	"strings"
	"time"
)

// ResourceRef identifies an API resource by group and plural resource name.
// Empty Namespace or Name fields act as wildcards.
type ResourceRef struct {
	Group     string // API group, e.g. "apps" ("" for the core group)
	Resource  string // plural resource name, e.g. "daemonsets"
	Namespace string // "" matches any namespace
	Name      string // "" matches any instance
}

// ChildResource describes a dependent resource managed on behalf of the root object.
type ChildResource struct {
	Group    string // API group of the child ("" for the core group)
	Resource string // plural resource name, e.g. "pods"

	// CreatedByController indicates the controller POSTs child objects through
	// the wrapped client, enabling write faults on child creation.
	CreatedByController bool
}

// ControllerProfile declares the resources and operations a controller exercises
// through the wrapped client. Scenarios inspect the profile so faults are only
// registered at sites the controller actually hits.
type ControllerProfile struct {
	Name string

	Root ResourceRef
	// WritesRoot and WritesRootStatus say which root writes arrive through the
	// wrapped client, whether the controller or the scenario action issues them.
	WritesRoot       bool // PUTs or PATCHes the root object
	WritesRootStatus bool // PUTs or PATCHes the root object's /status subresource

	Child            *ChildResource
	UsesExpectations bool // uses controller.ControllerExpectations

	// StaleCacheWindow is how long the stale-cache scenarios keep the
	// controller's informer caches from updating. Set it longer than any grace
	// period the controller applies. Zero means defaultStaleCacheWindow.
	StaleCacheWindow time.Duration

	// WriteFaultBurst is how many consecutive root writes WriteConflicts rejects.
	// Zero means 2.
	WriteFaultBurst int
}

// defaultStaleCacheWindow is long enough for a controller without a grace
// period to run several syncs against the stale cache.
const defaultStaleCacheWindow = 2 * time.Second

func (p ControllerProfile) staleCacheWindow() time.Duration {
	if p.StaleCacheWindow > 0 {
		return p.StaleCacheWindow
	}
	return defaultStaleCacheWindow
}

func (p ControllerProfile) writeFaultBurst() int {
	if p.WriteFaultBurst > 0 {
		return p.WriteFaultBurst
	}
	return 2
}

func (p ControllerProfile) validate() error {
	if p.Root.Resource == "" {
		return fmt.Errorf("ControllerProfile.Root.Resource must be set to the plural resource name (e.g. %q)", "daemonsets")
	}
	if strings.Contains(p.Root.Group, "/") {
		return fmt.Errorf("ControllerProfile.Root.Group must be the API group without version (e.g. %q, not %q)", strings.Split(p.Root.Group, "/")[0], p.Root.Group)
	}
	if p.Child != nil {
		if p.Child.Resource == "" {
			return fmt.Errorf("ControllerProfile.Child.Resource must be set to the plural resource name (e.g. %q)", "pods")
		}
		if strings.Contains(p.Child.Group, "/") {
			return fmt.Errorf("ControllerProfile.Child.Group must be the API group without version (e.g. %q, not %q)", strings.Split(p.Child.Group, "/")[0], p.Child.Group)
		}
	}
	return nil
}

// HasChildCache reports whether the controller has a child resource with a wrapped informer cache.
func (p ControllerProfile) HasChildCache() bool {
	return p.Child != nil && p.Child.Resource != ""
}

// CreatesChildren reports whether the controller creates child objects itself.
func (p ControllerProfile) CreatesChildren() bool {
	return p.Child != nil && p.Child.CreatedByController
}

// RootWriteMatch selects PUTs and PATCHes of the root object (excluding /status).
func (p ControllerProfile) RootWriteMatch() ClientMatch {
	return ClientMatch{
		Verbs:     []string{"PUT", "PATCH"},
		Group:     p.Root.Group,
		Resource:  p.Root.Resource,
		Namespace: p.Root.Namespace,
		Name:      p.Root.Name,
	}
}

// RootStatusWriteMatch selects PUTs and PATCHes of the root object's /status subresource.
func (p ControllerProfile) RootStatusWriteMatch() ClientMatch {
	m := p.RootWriteMatch()
	m.Subresource = "status"
	return m
}

// ChildCreateMatch selects POSTs of child objects.
func (p ControllerProfile) ChildCreateMatch() ClientMatch {
	return ClientMatch{Verbs: []string{"POST"}, Group: p.Child.Group, Resource: p.Child.Resource}
}

// ChildCacheMatch selects lookups in the child's wrapped informer cache.
func (p ControllerProfile) ChildCacheMatch() CacheMatch {
	return CacheMatch{Cache: p.Child.Resource}
}

// RootWatchMatch selects watch events for the root object.
func (p ControllerProfile) RootWatchMatch() ClientMatch {
	return ClientMatch{Verbs: []string{VerbWatch}, Group: p.Root.Group, Resource: p.Root.Resource, Namespace: p.Root.Namespace, Name: p.Root.Name}
}

// ChildWatchMatch selects watch events for child objects.
func (p ControllerProfile) ChildWatchMatch() ClientMatch {
	return ClientMatch{Verbs: []string{VerbWatch}, Group: p.Child.Group, Resource: p.Child.Resource}
}
