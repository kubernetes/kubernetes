/*
Copyright 2019 The Kubernetes Authors.

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

package resource

import (
	"errors"
	"fmt"

	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	yaml "gopkg.in/yaml.v2"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
)

// VerifyDryRun returns nil if a resource group-version-kind supports
// server-side dry-run. Otherwise, an error is returned.
func VerifyDryRun(gvk schema.GroupVersionKind, dynamicClient dynamic.Interface, discoveryClient discovery.DiscoveryInterface) error {
	verifier := NewDryRunVerifier(dynamicClient, discoveryClient)
	return verifier.HasSupport(gvk)
}

func NewDryRunVerifier(dynamicClient dynamic.Interface, discoveryClient discovery.DiscoveryInterface) *DryRunVerifier {
	return &DryRunVerifier{
		finder:        NewCRDFinder(CRDFromDynamic(dynamicClient)),
		openAPIGetter: discoveryClient,
	}
}

func hasGVKExtension(extensions []*openapi_v2.NamedAny, gvk schema.GroupVersionKind) bool {
	for _, extension := range extensions {
		if extension.GetValue().GetYaml() == "" ||
			extension.GetName() != "x-kubernetes-group-version-kind" {
			continue
		}
		var value map[string]string
		err := yaml.Unmarshal([]byte(extension.GetValue().GetYaml()), &value)
		if err != nil {
			continue
		}

		if value["group"] == gvk.Group && value["kind"] == gvk.Kind && value["version"] == gvk.Version {
			return true
		}
		return false
	}
	return false
}

// DryRunVerifier verifies if a given group-version-kind supports DryRun
// against the current server. Sending dryRun requests to apiserver that
// don't support it will result in objects being unwillingly persisted.
//
// It reads the OpenAPI to see if the given GVK supports dryRun. If the
// GVK can not be found, we assume that CRDs will have the same level of
// support as "namespaces", and non-CRDs will not be supported. We
// delay the check for CRDs as much as possible though, since it
// requires an extra round-trip to the server.
type DryRunVerifier struct {
	finder        CRDFinder
	openAPIGetter discovery.OpenAPISchemaInterface
}

// HasSupport verifies if the given gvk supports DryRun. An error is
// returned if it doesn't.
func (v *DryRunVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	oapi, err := v.openAPIGetter.OpenAPISchema()
	if err != nil {
		return fmt.Errorf("failed to download openapi: %v", err)
	}
	supports, err := supportsDryRun(oapi, gvk)
	if err != nil {
		// We assume that we couldn't find the type, then check for namespace:
		supports, _ = supportsDryRun(oapi, schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Namespace"})
		// If namespace supports dryRun, then we will support dryRun for CRDs only.
		if supports {
			supports, err = v.finder.HasCRD(gvk.GroupKind())
			if err != nil {
				return fmt.Errorf("failed to check CRD: %v", err)
			}
		}
	}
	if !supports {
		return fmt.Errorf("%v doesn't support dry-run", gvk)
	}
	return nil
}

// supportsDryRun is a method that let's us look in the OpenAPI if the
// specific group-version-kind supports the dryRun query parameter for
// the PATCH end-point.
func supportsDryRun(doc *openapi_v2.Document, gvk schema.GroupVersionKind) (bool, error) {
	for _, path := range doc.GetPaths().GetPath() {
		// Is this describing the gvk we're looking for?
		if !hasGVKExtension(path.GetValue().GetPatch().GetVendorExtension(), gvk) {
			continue
		}
		for _, param := range path.GetValue().GetPatch().GetParameters() {
			if param.GetParameter().GetNonBodyParameter().GetQueryParameterSubSchema().GetName() == "dryRun" {
				return true, nil
			}
		}
		return false, nil
	}

	return false, errors.New("couldn't find GVK in openapi")
}
