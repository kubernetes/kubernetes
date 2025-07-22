// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package accumulator

import (
	"encoding/json"
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/plugins/builtinconfig"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/filesys"
	"sigs.k8s.io/kustomize/kyaml/resid"
	"sigs.k8s.io/yaml"
)

// OpenAPIDefinition describes single type.
// Normally these definitions are auto-generated using gen-openapi.
// Same as in k8s.io / kube-openapi / pkg / common.
type OpenAPIDefinition struct {
	Schema       spec.Schema
	Dependencies []string
}

type myProperties = map[string]spec.Schema
type nameToApiMap map[string]OpenAPIDefinition

// LoadConfigFromCRDs parse CRD schemas from paths into a TransformerConfig
func LoadConfigFromCRDs(
	ldr ifc.Loader, paths []string) (*builtinconfig.TransformerConfig, error) {
	tc := builtinconfig.MakeEmptyConfig()
	for _, path := range paths {
		content, err := ldr.Load(path)
		if err != nil {
			return nil, err
		}
		m, err := makeNameToApiMap(content)
		if err != nil {
			return nil, errors.WrapPrefixf(err, "unable to parse open API definition from '%s'", path)
		}
		otherTc, err := makeConfigFromApiMap(m)
		if err != nil {
			return nil, err
		}
		tc, err = tc.Merge(otherTc)
		if err != nil {
			return nil, err
		}
	}
	return tc, nil
}

func makeNameToApiMap(content []byte) (result nameToApiMap, err error) {
	if content[0] == '{' {
		err = json.Unmarshal(content, &result)
	} else {
		err = yaml.Unmarshal(content, &result)
	}
	return
}

func makeConfigFromApiMap(m nameToApiMap) (*builtinconfig.TransformerConfig, error) {
	result := builtinconfig.MakeEmptyConfig()
	for name, api := range m {
		if !looksLikeAk8sType(api.Schema.SchemaProps.Properties) {
			continue
		}
		tc := builtinconfig.MakeEmptyConfig()
		err := loadCrdIntoConfig(
			tc, makeGvkFromTypeName(name), m, name, []string{})
		if err != nil {
			return result, err
		}
		result, err = result.Merge(tc)
		if err != nil {
			return result, err
		}
	}
	return result, nil
}

// TODO: Get Group and Version for CRD from the
// openAPI definition once
// "x-kubernetes-group-version-kind" is available in CRD
func makeGvkFromTypeName(n string) resid.Gvk {
	names := strings.Split(n, filesys.SelfDir)
	kind := names[len(names)-1]
	return resid.Gvk{Kind: kind}
}

func looksLikeAk8sType(properties myProperties) bool {
	_, ok := properties["kind"]
	if !ok {
		return false
	}
	_, ok = properties["apiVersion"]
	if !ok {
		return false
	}
	_, ok = properties["metadata"]
	return ok
}

const (
	// "x-kubernetes-annotation": ""
	xAnnotation = "x-kubernetes-annotation"

	// "x-kubernetes-label-selector": ""
	xLabelSelector = "x-kubernetes-label-selector"

	// "x-kubernetes-identity": ""
	xIdentity = "x-kubernetes-identity"

	// "x-kubernetes-object-ref-api-version": <apiVersion name>
	xVersion = "x-kubernetes-object-ref-api-version"

	// "x-kubernetes-object-ref-kind": <kind name>
	xKind = "x-kubernetes-object-ref-kind"

	// "x-kubernetes-object-ref-name-key": "name"
	// default is "name"
	xNameKey = "x-kubernetes-object-ref-name-key"
)

// loadCrdIntoConfig loads a CRD spec into a TransformerConfig
func loadCrdIntoConfig(
	theConfig *builtinconfig.TransformerConfig, theGvk resid.Gvk, theMap nameToApiMap,
	typeName string, path []string) (err error) {
	api, ok := theMap[typeName]
	if !ok {
		return nil
	}
	for propName, property := range api.Schema.SchemaProps.Properties {
		_, annotate := property.Extensions.GetString(xAnnotation)
		if annotate {
			err = theConfig.AddAnnotationFieldSpec(
				makeFs(theGvk, append(path, propName)))
			if err != nil {
				return
			}
		}
		_, label := property.Extensions.GetString(xLabelSelector)
		if label {
			err = theConfig.AddCommonLabelsFieldSpec(
				makeFs(theGvk, append(path, propName)))
			if err != nil {
				return
			}
		}
		_, identity := property.Extensions.GetString(xIdentity)
		if identity {
			err = theConfig.AddPrefixFieldSpec(
				makeFs(theGvk, append(path, propName)))
			if err != nil {
				return
			}
		}
		version, ok := property.Extensions.GetString(xVersion)
		if ok {
			kind, ok := property.Extensions.GetString(xKind)
			if ok {
				nameKey, ok := property.Extensions.GetString(xNameKey)
				if !ok {
					nameKey = "name"
				}
				err = theConfig.AddNamereferenceFieldSpec(
					builtinconfig.NameBackReferences{
						Gvk: resid.Gvk{Kind: kind, Version: version},
						Referrers: []types.FieldSpec{
							makeFs(theGvk, append(path, propName, nameKey))},
					})
				if err != nil {
					return
				}
			}
		}
		if property.Ref.GetURL() != nil {
			err = loadCrdIntoConfig(
				theConfig, theGvk, theMap,
				property.Ref.String(), append(path, propName))
			if err != nil {
				return
			}
		}
	}
	return nil
}

func makeFs(in resid.Gvk, path []string) types.FieldSpec {
	return types.FieldSpec{
		CreateIfNotPresent: false,
		Gvk:                in,
		Path:               strings.Join(path, "/"),
	}
}
