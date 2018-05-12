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

package v1alpha1

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/json-iterator/go"
	"github.com/ugorji/go/codec"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	runtime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

var json = jsoniter.ConfigCompatibleWithStandardLibrary

type configMutationFunc func(map[string]interface{}) error

// These migrations are a stop-gap until we get a properly-versioned configuration file for MasterConfiguration.
// https://github.com/kubernetes/kubeadm/issues/750
var migrations = map[string][]configMutationFunc{
	"MasterConfiguration": {
		proxyFeatureListToMap,
	},
}

// Migrate takes a map representing a config file and an object to decode into.
// The map is transformed into a format suitable for encoding into the supplied object, then serialised and decoded.
func Migrate(in map[string]interface{}, obj runtime.Object) error {
	kind := reflect.TypeOf(obj).Elem().Name()
	migrationsForKind := migrations[kind]

	for _, m := range migrationsForKind {
		err := m(in)
		if err != nil {
			return err
		}
	}

	// Use codec instead of encoding/json to handle map[interface{}]interface{}
	handle := &codec.JsonHandle{}
	buf := new(bytes.Buffer)
	if err := codec.NewEncoder(buf, handle).Encode(in); err != nil {
		return fmt.Errorf("couldn't json encode object: %v", err)
	}

	return runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), buf.Bytes(), obj)
}

func proxyFeatureListToMap(m map[string]interface{}) error {
	featureGatePath := []string{"kubeProxy", "config", "featureGates"}

	// If featureGatePath is already a map, we don't need to do anything.
	_, _, err := unstructured.NestedMap(m, featureGatePath...)
	if err == nil {
		return nil
	}

	gates, _, err := unstructured.NestedString(m, featureGatePath...)
	if err != nil {
		return fmt.Errorf("couldn't get featureGates: %v", err)
	}

	gateMap := make(map[string]interface{})
	for _, gate := range strings.Split(gates, ",") {
		if gate == "" {
			continue
		}
		parts := strings.SplitN(gate, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("unparsable kubeproxy feature gate %q", gate)
		}
		val, err := strconv.ParseBool(parts[1])
		if err != nil {
			return fmt.Errorf("unparsable kubeproxy feature gate %q: %v", gate, err)
		}
		gateMap[parts[0]] = val
	}

	unstructured.SetNestedMap(m, gateMap, featureGatePath...)
	return nil
}

// LoadYAML is a small wrapper around go-yaml that ensures all nested structs are map[string]interface{} instead of map[interface{}]interface{}.
func LoadYAML(bytes []byte) (map[string]interface{}, error) {
	var decoded map[interface{}]interface{}
	if err := yaml.Unmarshal(bytes, &decoded); err != nil {
		return map[string]interface{}{}, fmt.Errorf("couldn't unmarshal YAML: %v", err)
	}

	converted, ok := convert(decoded).(map[string]interface{})
	if !ok {
		return map[string]interface{}{}, errors.New("yaml is not a map")
	}

	return converted, nil
}

// https://stackoverflow.com/questions/40737122/convert-yaml-to-json-without-struct-golang
func convert(i interface{}) interface{} {
	switch x := i.(type) {
	case map[interface{}]interface{}:
		m2 := map[string]interface{}{}
		for k, v := range x {
			m2[k.(string)] = convert(v)
		}
		return m2
	case []interface{}:
		for i, v := range x {
			x[i] = convert(v)
		}
	}
	return i
}
