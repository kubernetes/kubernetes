// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resmap

import (
	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/kusterr"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Factory makes instances of ResMap.
type Factory struct {
	// Makes resources.
	resF *resource.Factory
}

// NewFactory returns a new resmap.Factory.
func NewFactory(rf *resource.Factory) *Factory {
	return &Factory{resF: rf}
}

// RF returns a resource.Factory.
func (rmF *Factory) RF() *resource.Factory {
	return rmF.resF
}

func New() ResMap {
	return newOne()
}

// FromResource returns a ResMap with one entry.
func (rmF *Factory) FromResource(res *resource.Resource) ResMap {
	m, err := newResMapFromResourceSlice([]*resource.Resource{res})
	if err != nil {
		panic(err)
	}
	return m
}

// FromResourceSlice returns a ResMap with a slice of resources.
func (rmF *Factory) FromResourceSlice(ress []*resource.Resource) ResMap {
	m, err := newResMapFromResourceSlice(ress)
	if err != nil {
		panic(err)
	}
	return m
}

// FromFile returns a ResMap given a resource path.
func (rmF *Factory) FromFile(
	loader ifc.Loader, path string) (ResMap, error) {
	content, err := loader.Load(path)
	if err != nil {
		return nil, err
	}
	m, err := rmF.NewResMapFromBytes(content)
	if err != nil {
		return nil, kusterr.Handler(err, path)
	}
	return m, nil
}

// NewResMapFromBytes decodes a list of objects in byte array format.
func (rmF *Factory) NewResMapFromBytes(b []byte) (ResMap, error) {
	resources, err := rmF.resF.SliceFromBytes(b)
	if err != nil {
		return nil, err
	}
	return newResMapFromResourceSlice(resources)
}

// NewResMapFromConfigMapArgs returns a Resource slice given
// a configmap metadata slice from kustomization file.
func (rmF *Factory) NewResMapFromConfigMapArgs(
	kvLdr ifc.KvLoader, argList []types.ConfigMapArgs) (ResMap, error) {
	var resources []*resource.Resource
	for i := range argList {
		res, err := rmF.resF.MakeConfigMap(kvLdr, &argList[i])
		if err != nil {
			return nil, errors.Wrap(err, "NewResMapFromConfigMapArgs")
		}
		resources = append(resources, res)
	}
	return newResMapFromResourceSlice(resources)
}

// FromConfigMapArgs creates a new ResMap containing one ConfigMap.
func (rmF *Factory) FromConfigMapArgs(
	kvLdr ifc.KvLoader, args types.ConfigMapArgs) (ResMap, error) {
	res, err := rmF.resF.MakeConfigMap(kvLdr, &args)
	if err != nil {
		return nil, err
	}
	return rmF.FromResource(res), nil
}

// NewResMapFromSecretArgs takes a SecretArgs slice, generates
// secrets from each entry, and accumulates them in a ResMap.
func (rmF *Factory) NewResMapFromSecretArgs(
	kvLdr ifc.KvLoader, argsList []types.SecretArgs) (ResMap, error) {
	var resources []*resource.Resource
	for i := range argsList {
		res, err := rmF.resF.MakeSecret(kvLdr, &argsList[i])
		if err != nil {
			return nil, errors.Wrap(err, "NewResMapFromSecretArgs")
		}
		resources = append(resources, res)
	}
	return newResMapFromResourceSlice(resources)
}

// FromSecretArgs creates a new ResMap containing one secret.
func (rmF *Factory) FromSecretArgs(
	kvLdr ifc.KvLoader, args types.SecretArgs) (ResMap, error) {
	res, err := rmF.resF.MakeSecret(kvLdr, &args)
	if err != nil {
		return nil, err
	}
	return rmF.FromResource(res), nil
}

func newResMapFromResourceSlice(
	resources []*resource.Resource) (ResMap, error) {
	result := New()
	for _, res := range resources {
		err := result.Append(res)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// NewResMapFromRNodeSlice returns a ResMap from a slice of RNodes
func (rmF *Factory) NewResMapFromRNodeSlice(s []*yaml.RNode) (ResMap, error) {
	rs, err := rmF.resF.ResourcesFromRNodes(s)
	if err != nil {
		return nil, err
	}
	return newResMapFromResourceSlice(rs)
}
