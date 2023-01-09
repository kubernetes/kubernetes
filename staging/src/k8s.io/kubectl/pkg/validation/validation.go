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

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kube-openapi/pkg/util/proto/validation"
	"k8s.io/kubectl/pkg/util/openapi"
)

// schemaValidation validates the object against an OpenAPI schema.
type schemaValidation struct {
	resources openapi.Resources
}

// NewSchemaValidation creates a new Schema that can be used
// to validate objects.
func NewSchemaValidation(resources openapi.Resources) Schema {
	return &schemaValidation{
		resources: resources,
	}
}

// ValidateBytes will validates the object against using the Resources
// object.
func (v *schemaValidation) ValidateBytes(data []byte) error {
	obj, err := parse(data)
	if err != nil {
		return err
	}

	gvk, errs := getObjectKind(obj)
	if errs != nil {
		return utilerrors.NewAggregate(errs)
	}

	if (gvk == schema.GroupVersionKind{Version: "v1", Kind: "List"}) {
		return utilerrors.NewAggregate(v.validateList(obj))
	}

	return utilerrors.NewAggregate(v.validateResource(obj, gvk))
}

func (v *schemaValidation) validateList(object interface{}) []error {
	fields, ok := object.(map[string]interface{})
	if !ok || fields == nil {
		return []error{errors.New("invalid object to validate")}
	}

	allErrors := []error{}
	if _, ok := fields["items"].([]interface{}); !ok {
		return []error{errors.New("invalid object to validate")}
	}
	for _, item := range fields["items"].([]interface{}) {
		if gvk, errs := getObjectKind(item); errs != nil {
			allErrors = append(allErrors, errs...)
		} else {
			allErrors = append(allErrors, v.validateResource(item, gvk)...)
		}
	}
	return allErrors
}

func (v *schemaValidation) validateResource(obj interface{}, gvk schema.GroupVersionKind) []error {
	resource := v.resources.LookupResource(gvk)
	if resource == nil {
		// resource is not present, let's just skip validation.
		return nil
	}

	return validation.ValidateModel(obj, resource, gvk.Kind)
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

func getObjectKind(object interface{}) (schema.GroupVersionKind, []error) {
	var listErrors []error
	fields, ok := object.(map[string]interface{})
	if !ok || fields == nil {
		listErrors = append(listErrors, errors.New("invalid object to validate"))
		return schema.GroupVersionKind{}, listErrors
	}

	var group string
	var version string
	apiVersion := fields["apiVersion"]
	if apiVersion == nil {
		listErrors = append(listErrors, errors.New("apiVersion not set"))
	} else if _, ok := apiVersion.(string); !ok {
		listErrors = append(listErrors, errors.New("apiVersion isn't string type"))
	} else {
		gv, err := schema.ParseGroupVersion(apiVersion.(string))
		if err != nil {
			listErrors = append(listErrors, err)
		} else {
			group = gv.Group
			version = gv.Version
		}
	}
	kind := fields["kind"]
	if kind == nil {
		listErrors = append(listErrors, errors.New("kind not set"))
	} else if _, ok := kind.(string); !ok {
		listErrors = append(listErrors, errors.New("kind isn't string type"))
	}
	if listErrors != nil {
		return schema.GroupVersionKind{}, listErrors
	}

	return schema.GroupVersionKind{Group: group, Version: version, Kind: kind.(string)}, nil
}
