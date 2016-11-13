/*
Copyright 2014 The Kubernetes Authors.

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

package initialization

import (
	"fmt"
	"io"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/auth/authorizer"
)

func init() {
	admission.RegisterPlugin("Initializers", func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		return NewInitializer(), nil
	})
}

type initializerOptions struct {
	Initializers []string
}

type initializer struct {
	resources  map[unversioned.GroupResource]initializerOptions
	authorizer authorizer.Authorizer
}

// NewAlwaysAdmit creates a new always admit admission handler
func NewInitializer() admission.Interface {
	return &initializer{
		resources: map[unversioned.GroupResource]initializerOptions{
			unversioned.GroupResource{Resource: "pods"}: {Initializers: []string{"Test"}},
		},
	}
}

func (i *initializer) Validate() error {
	if i.authorizer == nil {
		return fmt.Errorf("requires authorizer")
	}
	return nil
}

func (i *initializer) SetAuthorizer(a authorizer.Authorizer) {
	i.authorizer = a
}

func (i *initializer) Admit(a admission.Attributes) (err error) {
	// TODO: sub-resource action should be denied until the object is initialized
	if len(a.GetSubresource()) > 0 {
		return nil
	}

	resource, ok := i.resources[a.GetResource().GroupResource()]
	if !ok {
		return nil
	}

	switch a.GetOperation() {
	case admission.Create:
		accessor, err := meta.Accessor(a.GetObject())
		if err != nil {
			return fmt.Errorf("initialized resources must be able to set initializers (%T): %v", a.GetObject(), err)
		}
		existing := accessor.GetInitializers()
		// for now, disallow sending explicit initializers
		if len(existing) > 0 {
			return fmt.Errorf("initializers may not be set on creation")
		}
		// it must be possible for some users to bypass initialization - for now, check the initialize operation
		// if the user provides an empty array for initialization (vs a nil one)
		if existing != nil {
			if err := i.canInitialize(a); err != nil {
				return err
			}
		}
		accessor.SetInitializers(resource.Initializers)

	case admission.Update:
		accessor, err := meta.Accessor(a.GetObject())
		if err != nil {
			return fmt.Errorf("initialized resources must be able to set initializers (%T): %v", a.GetObject(), err)
		}
		existing := accessor.GetInitializers()

		// post initialization, all changes are allowed
		if len(existing) == 0 {
			return nil
		}

		// caller must have the ability to mutate un-initialized resources
		if err := i.canInitialize(a); err != nil {
			return err
		}

		// TODO: restrict initialization list changes to specific clients?

		// TODO: reject mutation of the object after initialization failure status is set?
	}

	return nil
}

func (i *initializer) canInitialize(a admission.Attributes) error {
	// caller must have the ability to mutate un-initialized resources
	authorized, reason, err := i.authorizer.Authorize(authorizer.AttributesRecord{
		Name:            a.GetName(),
		ResourceRequest: true,
		User:            a.GetUserInfo(),
		Verb:            "initialize",
		Namespace:       a.GetNamespace(),
		APIGroup:        a.GetResource().Group,
		APIVersion:      a.GetResource().Version,
		Resource:        a.GetResource().Resource,
	})
	if err != nil {
		return err
	}
	if !authorized {
		return fmt.Errorf("user must have permission to initialize resources: %s", reason)
	}
	return nil
}

func (i *initializer) Handles(op admission.Operation) bool {
	return true
}
