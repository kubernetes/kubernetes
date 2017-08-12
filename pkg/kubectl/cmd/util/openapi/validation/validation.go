/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/yaml"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

type SchemaValidation struct {
	resources openapi.Resources
}

func NewSchemaValidation(resources openapi.Resources) *SchemaValidation {
	return &SchemaValidation{
		resources: resources,
	}
}

func (v *SchemaValidation) Validate(data []byte) error {
	obj, err := parse(data)
	if err != nil {
		return err
	}

	gvk, err := getObjectKind(obj)
	if err != nil {
		return err
	}

	resource := v.resources.LookupResource(gvk)
	if resource == nil {
		return fmt.Errorf("unknown object type %q", gvk)
	}

	rootValidation, err := itemFactory(openapi.NewPath(gvk.Kind), obj)
	if err != nil {
		return err
	}
	resource.Accept(rootValidation)
	errs := rootValidation.Errors()
	if errs != nil {
		return utilerrors.NewAggregate(errs)
	}
	return nil
}

func parse(data []byte) (interface{}, error) {
	var obj interface{}
	out, err := yaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(out, &obj); err != nil {
		return nil, err
	}
	return obj, nil
}

func getObjectKind(object interface{}) (schema.GroupVersionKind, error) {
	fields := object.(map[string]interface{})
	if fields == nil {
		return schema.GroupVersionKind{}, errors.New("invalid object to validate")
	}
	apiVersion := fields["apiVersion"]
	if apiVersion == nil {
		return schema.GroupVersionKind{}, errors.New("apiVersion not set")
	}
	if _, ok := apiVersion.(string); !ok {
		return schema.GroupVersionKind{}, errors.New("apiVersion isn't string type")
	}
	version := apiutil.GetVersion(apiVersion.(string))
	kind := fields["kind"]
	if kind == nil {
		return schema.GroupVersionKind{}, errors.New("kind not set")
	}
	if _, ok := kind.(string); !ok {
		return schema.GroupVersionKind{}, errors.New("kind isn't string type")
	}

	return schema.GroupVersionKind{Kind: kind.(string), Version: version}, nil
}
