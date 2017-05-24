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

package nodeconfig

import (
	"encoding/json"
	"fmt"

	yaml "k8s.io/apimachinery/pkg/util/yaml"
	api "k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	ccv1a1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
)

const configMapConfigKey = "kubelet"

// parsable is for parsing a KubeletConfiguration out of a config object
type parsable interface {
	parse() (*ccv1a1.KubeletConfiguration, error)
}

// parsableConfigMap is for parsing a KubeletConfiguration out of a ConfigMap
type parsableConfigMap struct {
	cm *apiv1.ConfigMap
}

// parse deserializes the KubeletConfiguration out of the ConfigMap's Data, and also applies defaults.
// If parsing fails, returns an error.
// If parsing succeeds, returns a KubeletConfiguration.
// It is recommended that you validate any returned configuration before using it.
func (p *parsableConfigMap) parse() (*ccv1a1.KubeletConfiguration, error) {
	const emptyCfgErr = "configuration was empty, but some parameters are required"
	if len(p.cm.Data) == 0 {
		return nil, fmt.Errorf(emptyCfgErr)
	}

	// TODO(mtaufen): Once the KubeletConfiguration type is decomposed
	v, ok := p.cm.Data[configMapConfigKey]
	if !ok {
		return nil, fmt.Errorf("key %q not found in ConfigMap", configMapConfigKey)
	} else if len(v) == 0 {
		return nil, fmt.Errorf(emptyCfgErr)
	}

	j, err := yaml.ToJSON([]byte(v))
	if err != nil {
		return nil, err
	}
	kc := &ccv1a1.KubeletConfiguration{}
	if err := json.Unmarshal(j, kc); err != nil {
		return nil, err
	}

	// run the defaulter on the loaded configuration before returning
	api.Scheme.Default(kc)

	return kc, nil
}
