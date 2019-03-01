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

package kunstruct

import (
	"bytes"
	"fmt"
	"io"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/configmapandsecret"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/types"
)

// KunstructuredFactoryImpl hides construction using apimachinery types.
type KunstructuredFactoryImpl struct {
	cmFactory     *configmapandsecret.ConfigMapFactory
	secretFactory *configmapandsecret.SecretFactory
}

var _ ifc.KunstructuredFactory = &KunstructuredFactoryImpl{}

// NewKunstructuredFactoryImpl returns a factory.
func NewKunstructuredFactoryImpl() ifc.KunstructuredFactory {
	return &KunstructuredFactoryImpl{}
}

// SliceFromBytes returns a slice of Kunstructured.
func (kf *KunstructuredFactoryImpl) SliceFromBytes(
	in []byte) ([]ifc.Kunstructured, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader(in), 1024)
	var result []ifc.Kunstructured
	var err error
	for err == nil || isEmptyYamlError(err) {
		var out unstructured.Unstructured
		err = decoder.Decode(&out)
		if err == nil {
			if len(out.Object) == 0 {
				continue
			}
			err = kf.validate(out)
			if err != nil {
				return nil, err
			}
			result = append(result, &UnstructAdapter{Unstructured: out})
		}
	}
	if err != io.EOF {
		return nil, err
	}
	return result, nil
}

func isEmptyYamlError(err error) bool {
	return strings.Contains(err.Error(), "is missing in 'null'")
}

// FromMap returns an instance of Kunstructured.
func (kf *KunstructuredFactoryImpl) FromMap(
	m map[string]interface{}) ifc.Kunstructured {
	return &UnstructAdapter{Unstructured: unstructured.Unstructured{Object: m}}
}

// MakeConfigMap returns an instance of Kunstructured for ConfigMap
func (kf *KunstructuredFactoryImpl) MakeConfigMap(args *types.ConfigMapArgs, options *types.GeneratorOptions) (ifc.Kunstructured, error) {
	cm, err := kf.cmFactory.MakeConfigMap(args, options)
	if err != nil {
		return nil, err
	}
	return NewKunstructuredFromObject(cm)
}

// MakeSecret returns an instance of Kunstructured for Secret
func (kf *KunstructuredFactoryImpl) MakeSecret(args *types.SecretArgs, options *types.GeneratorOptions) (ifc.Kunstructured, error) {
	sec, err := kf.secretFactory.MakeSecret(args, options)
	if err != nil {
		return nil, err
	}
	return NewKunstructuredFromObject(sec)
}

// Set sets loader
func (kf *KunstructuredFactoryImpl) Set(ldr ifc.Loader) {
	kf.cmFactory = configmapandsecret.NewConfigMapFactory(ldr)
	kf.secretFactory = configmapandsecret.NewSecretFactory(ldr)
}

// validate validates that u has kind and name
// except for kind `List`, which doesn't require a name
func (kf *KunstructuredFactoryImpl) validate(u unstructured.Unstructured) error {
	kind := u.GetKind()
	if kind == "" {
		return fmt.Errorf("missing kind in object %v", u)
	} else if kind == "List" {
		return nil
	}
	if u.GetName() == "" {
		return fmt.Errorf("missing metadata.name in object %v", u)
	}
	return nil
}
