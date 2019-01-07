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

package config

import (
	"encoding/json"
	"strings"

	"github.com/ghodss/yaml"
	"k8s.io/kube-openapi/pkg/common"
	"sigs.k8s.io/kustomize/pkg/gvk"
)

// LoadCRDs parse CRD schemas from paths into a TransformerConfig
func (tf *Factory) LoadCRDs(paths []string) (*TransformerConfig, error) {
	tc := tf.EmptyConfig()
	for _, path := range paths {
		otherTc, err := tf.loadCRD(path)
		if err != nil {
			return nil, err
		}
		tc = tc.Merge(otherTc)
	}
	return tc, nil
}

func (tf *Factory) loadCRD(path string) (*TransformerConfig, error) {
	result := tf.EmptyConfig()
	content, err := tf.loader().Load(path)
	if err != nil {
		return result, err
	}

	var types map[string]common.OpenAPIDefinition
	if content[0] == '{' {
		err = json.Unmarshal(content, &types)
	} else {
		err = yaml.Unmarshal(content, &types)
	}
	if err != nil {
		return nil, err
	}

	crds := getCRDs(types)
	for crd, k := range crds {
		tc := tf.EmptyConfig()
		err = loadCrdIntoConfig(
			types, crd, crd, k, []string{}, tc)
		if err != nil {
			return result, err
		}
		result = result.Merge(tc)
	}

	return result, nil
}

// getCRDs get all CRD types
func getCRDs(types map[string]common.OpenAPIDefinition) map[string]gvk.Gvk {
	crds := map[string]gvk.Gvk{}

	for typename, t := range types {
		properties := t.Schema.SchemaProps.Properties
		_, foundKind := properties["kind"]
		_, foundAPIVersion := properties["apiVersion"]
		_, foundMetadata := properties["metadata"]
		if foundKind && foundAPIVersion && foundMetadata {
			// TODO: Get Group and Version for CRD from the openAPI definition once
			// "x-kubernetes-group-version-kind" is available in CRD
			kind := strings.Split(typename, ".")[len(strings.Split(typename, "."))-1]
			crds[typename] = gvk.Gvk{Kind: kind}
		}
	}
	return crds
}

// loadCrdIntoConfig loads a CRD spec into a TransformerConfig
func loadCrdIntoConfig(
	types map[string]common.OpenAPIDefinition,
	atype string, crd string, in gvk.Gvk,
	path []string, config *TransformerConfig) error {
	if _, ok := types[crd]; !ok {
		return nil
	}

	for propname, property := range types[atype].Schema.SchemaProps.Properties {
		_, annotate := property.Extensions.GetString(Annotation)
		if annotate {
			config.AddAnnotationFieldSpec(
				FieldSpec{
					CreateIfNotPresent: false,
					Gvk:                in,
					Path:               strings.Join(append(path, propname), "/"),
				},
			)
		}
		_, label := property.Extensions.GetString(LabelSelector)
		if label {
			config.AddLabelFieldSpec(
				FieldSpec{
					CreateIfNotPresent: false,
					Gvk:                in,
					Path:               strings.Join(append(path, propname), "/"),
				},
			)
		}
		_, identity := property.Extensions.GetString(Identity)
		if identity {
			config.AddPrefixFieldSpec(
				FieldSpec{
					CreateIfNotPresent: false,
					Gvk:                in,
					Path:               strings.Join(append(path, propname), "/"),
				},
			)
		}
		version, ok := property.Extensions.GetString(Version)
		if ok {
			kind, ok := property.Extensions.GetString(Kind)
			if ok {
				nameKey, ok := property.Extensions.GetString(NameKey)
				if !ok {
					nameKey = "name"
				}
				config.AddNamereferenceFieldSpec(NameBackReferences{
					Gvk: gvk.Gvk{Kind: kind, Version: version},
					FieldSpecs: []FieldSpec{
						{
							CreateIfNotPresent: false,
							Gvk:                in,
							Path:               strings.Join(append(path, propname, nameKey), "/"),
						},
					},
				})
			}
		}

		if property.Ref.GetURL() != nil {
			loadCrdIntoConfig(
				types, property.Ref.String(), crd, in,
				append(path, propname), config)
		}
	}
	return nil
}
