/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/ghodss/yaml"
	"k8s.io/kubernetes/pkg/runtime"
)

// GeneratorTransformer is a StreamTransform that knows how to transform a YAML file with parameters
// into an API object via a Generator interface.
// For example:
//   params.yaml looks like:
//    name: foo
//    replicas: 2
//    image: foo
//
// and the resulting created replication controller created by the 'v1/run' generator looks like:
//
//  apiVersion: v1
//  kind: ReplicationController
//  metadata:
//  labels:
//    run: foo
//  name: foo
//  namespace: default
//  spec:
//    replicas: 2
//    selector:
//      run: foo
//    template:
//      metadata:
//        creationTimestamp: null
//        labels:
//          run: foo
//   ...
type GeneratorTransformer struct {
	Generator Generator
	Codec     runtime.Codec
}

// Transform implements the StreamTransform interface
func (g *GeneratorTransformer) Transform(in io.Reader) (io.Reader, error) {
	data, err := ioutil.ReadAll(in)
	if err != nil {
		return nil, err
	}
	var obj interface{}
	if err = yaml.Unmarshal(data, &obj); err != nil {
		return nil, err
	}
	mapObj, ok := obj.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected type: %v", obj)
	}
	runtimeObj, err := g.Generator.Generate(mapObj)
	if err != nil {
		return nil, err
	}

	if data, err = g.Codec.Encode(runtimeObj); err != nil {
		return nil, err
	}

	return bytes.NewBuffer(data), nil
}
