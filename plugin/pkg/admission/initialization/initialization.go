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

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("Initializers", func(config io.Reader) (admission.Interface, error) {
		return NewInitializer(), nil
	})
}

type initializerOptions struct {
	Initializers []string
}

type initializer struct {
	resources  map[schema.GroupResource]initializerOptions
	authorizer authorizer.Authorizer
}

// NewAlwaysAdmit creates a new always admit admission handler
func NewInitializer() admission.Interface {
	return &initializer{
		resources: map[schema.GroupResource]initializerOptions{
		//schema.GroupResource{Resource: "pods"}: {Initializers: []string{"Test"}},
		},
	}
}

func (i *initializer) Validate() error {
	return nil
}

func (i *initializer) SetAuthorizer(a authorizer.Authorizer) {
	i.authorizer = a
}

var initializerFieldPath = field.NewPath("metadata", "initializers")

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
			// objects without meta accessor cannot be checked for initialization, and it is possible to make calls
			// via our API that don't have ObjectMeta
			return nil
		}
		existing := accessor.GetInitializers()
		// it must be possible for some users to bypass initialization - for now, check the initialize operation
		if existing != nil {
			if err := i.canInitialize(a); err != nil {
				return err
			}
		}

		// TODO: pull this from config
		accessor.SetInitializers(copiedInitializers(resource.Initializers))

	case admission.Update:
		accessor, err := meta.Accessor(a.GetObject())
		if err != nil {
			// objects without meta accessor cannot be checked for initialization, and it is possible to make calls
			// via our API that don't have ObjectMeta
			return nil
		}
		updated := accessor.GetInitializers()

		existingAccessor, err := meta.Accessor(a.GetOldObject())
		if err != nil {
			// if the old object does not have an accessor, but the new one does, error out
			return fmt.Errorf("initialized resources must be able to set initializers (%T): %v", a.GetOldObject(), err)
		}
		existing := existingAccessor.GetInitializers()

		// because we are called before validation, we need to ensure the update transition is valid.
		if errs := validation.ValidateInitializersUpdate(updated, existing, initializerFieldPath); len(errs) > 0 {
			return errors.NewInvalid(a.GetKind().GroupKind(), a.GetName(), errs)
		}

		// caller must have the ability to mutate un-initialized resources
		if err := i.canInitialize(a); err != nil {
			return err
		}

		// TODO: restrict initialization list changes to specific clients?
	}

	return nil
}

func (i *initializer) canInitialize(a admission.Attributes) error {
	// if no authorizer is present, the initializer plugin allows modification of uninitialized resources
	if i.authorizer == nil {
		glog.V(4).Infof("No authorizer provided to initialization admission control, unable to check permissions")
		return nil
	}
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

func copiedInitializers(names []string) *metav1.Initializers {
	if len(names) == 0 {
		return nil
	}
	var init []metav1.Initializer
	for _, name := range names {
		init = append(init, metav1.Initializer{Name: name})
	}
	return &metav1.Initializers{
		Pending: init,
	}
}
