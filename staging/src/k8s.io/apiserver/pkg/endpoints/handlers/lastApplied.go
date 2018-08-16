/*
Copyright 2018 The Kubernetes Authors.

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

package handlers

import (
	"fmt"

	"github.com/ghodss/yaml"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
)

// TODO(apelisse): workflowId needs to be passed as a query
// param/header, and a better defaulting needs to be defined too.
const workflowID = "default"

// LastAppliedAccessor allows saving and extracting intents to objects
type LastAppliedAccessor interface {
	New(data []byte) (map[string]interface{}, error)
	Save(data map[string]interface{}, obj runtime.Object) error
	SaveNew(data []byte, obj runtime.Object) error
	Extract(obj runtime.Object) (map[string]interface{}, error)
}

type lastAppliedAccessor struct {
	workflow string
	kind     schema.GroupVersionKind
}

// NewLastAppliedAccessor takes a workflowID and an object kind and pointer to access
func NewLastAppliedAccessor(
	workflow string,
	kind schema.GroupVersionKind,
) LastAppliedAccessor {
	return &lastAppliedAccessor{
		workflow: workflow,
		kind:     kind,
	}
}

func (a *lastAppliedAccessor) New(data []byte) (map[string]interface{}, error) {
	intent := make(map[string]interface{})
	if err := yaml.Unmarshal(data, &intent); err != nil {
		return nil, fmt.Errorf("couldn't unmarshal object: %v (data: %v)", err, string(data))
	}
	return intent, nil
}

func (a *lastAppliedAccessor) Save(intent map[string]interface{}, obj runtime.Object) error {
	// Make sure we have the gvk set on the object.
	(&unstructured.Unstructured{Object: intent}).SetGroupVersionKind(a.kind)

	j, err := json.Marshal(intent)
	if err != nil {
		return fmt.Errorf("failed to serialize json: %v", err)
	}

	accessor, err := meta.Accessor(obj)
	if err != nil {
		return fmt.Errorf("couldn't get accessor: %v", err)
	}

	m := accessor.GetLastApplied()
	if m == nil {
		m = make(map[string]runtime.RawExtension)
	}

	m[a.workflow] = runtime.RawExtension{Raw: j}
	accessor.SetLastApplied(m)

	return nil
}

func (a *lastAppliedAccessor) SaveNew(data []byte, obj runtime.Object) error {
	intent, err := a.New(data)
	if err != nil {
		return err
	}
	err = a.Save(intent, obj)
	return err
}

func (a *lastAppliedAccessor) Extract(obj runtime.Object) (map[string]interface{}, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}
	last := make(map[string]interface{})
	if _, ok := accessor.GetLastApplied()[a.workflow]; !ok {
		// TODO: fix this
		last, err = runtime.DefaultUnstructuredConverter.ToUnstructured(accessor.GetLastApplied()[a.workflow])
		if err != nil {
			return nil, fmt.Errorf("couldn't convert lastApplied: %v", err)
		}
	}
	return last, nil
}
