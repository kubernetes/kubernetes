/*
Copyright 2022 The Kubernetes Authors.

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

package flowcontrol

import (
	"fmt"
	"io/ioutil"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/apis/apiserver/install"
	"k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
	fcrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
	"sigs.k8s.io/yaml"
)

var cfgScheme = runtime.NewScheme()

func init() {
	install.Install(cfgScheme)
}

// DefaultConfig creates a new PriorityAndFairnessConfiguration with default values.
func DefaultConfig() apiserver.PriorityAndFairnessConfiguration {
	return apiserver.PriorityAndFairnessConfiguration{
		WorkEstimator: fcrequest.DefaultWorkEstimatorConfiguration(),
	}
}

// ReadConfigFromFile parses configuration stored under configFilePath
// and overrides PriorityAndFairnessConfiguration cfg object with defined values,
// leaving undefined ones unchanged.
func ReadConfigFromFile(configFilePath string, cfg *apiserver.PriorityAndFairnessConfiguration) error {
	if configFilePath == "" {
		return nil
	}

	data, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return fmt.Errorf("unable to read P&F configuration from %q [%v]", configFilePath, err)
	}

	decodedConfig := v1alpha1.PriorityAndFairnessConfiguration{}
	if err := cfgScheme.Convert(cfg, &decodedConfig, nil); err != nil {
		return err
	}

	err = yaml.Unmarshal(data, &decodedConfig)
	if err != nil {
		return err
	}

	if decodedConfig.Kind != "PriorityAndFairnessConfiguration" {
		return fmt.Errorf("invalid service configuration object %q", decodedConfig.Kind)
	}

	if err := cfgScheme.Convert(&decodedConfig, cfg, nil); err != nil {
		return err
	}

	return nil
}
