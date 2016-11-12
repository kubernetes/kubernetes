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
	resources map[unversioned.GroupResource]initializerOptions
}

// NewAlwaysAdmit creates a new always admit admission handler
func NewInitializer() admission.Interface {
	return initializer{
		resources: map[unversioned.GroupResource]initializerOptions{
			unversioned.GroupResource{Resource: "pods"}: {Initializers: []string{"Test"}},
		},
	}
}

func (i initializer) Admit(a admission.Attributes) (err error) {
	if !i.Handles(a.GetOperation()) {
		return nil
	}
	if len(a.GetSubresource()) > 0 {
		return nil
	}
	resource, ok := i.resources[a.GetResource().GroupResource()]
	if !ok {
		return nil
	}
	fmt.Printf("Setting %v on %v\n", resource.Initializers, a.GetResource())
	accessor, err := meta.Accessor(a.GetObject())
	if err != nil {
		return fmt.Errorf("initialized resources must be able to set initializers (%T): %v", a.GetObject(), err)
	}
	if existing := accessor.GetInitializers(); len(existing) > 0 {
		return fmt.Errorf("initializers may not be set on creation")
	}
	accessor.SetInitializers(resource.Initializers)

	return nil
}

func (i initializer) Handles(op admission.Operation) bool {
	return op == admission.Create
}
