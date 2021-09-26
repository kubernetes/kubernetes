// +build go1.7

package vmutils

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
)

func updateOrAddConfig(configs []vm.ConfigurationSet, configType vm.ConfigurationSetType, update func(*vm.ConfigurationSet)) []vm.ConfigurationSet {
	config := findConfig(configs, configType)
	if config == nil {
		configs = append(configs, vm.ConfigurationSet{ConfigurationSetType: configType})
		config = findConfig(configs, configType)
	}
	update(config)

	return configs
}

func findConfig(configs []vm.ConfigurationSet, configType vm.ConfigurationSetType) *vm.ConfigurationSet {
	for i, config := range configs {
		if config.ConfigurationSetType == configType {
			// need to return a pointer to the original set in configs,
			// not the copy made by the range iterator
			return &configs[i]
		}
	}

	return nil
}
