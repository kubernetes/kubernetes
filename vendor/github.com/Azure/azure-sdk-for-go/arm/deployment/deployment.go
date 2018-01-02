package deployment

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/json"
	"io/ioutil"

	"github.com/Azure/azure-sdk-for-go/arm/resources/resources"
)

// Build is a helper that creates a resources.Deployment, which can
// be used as a parameter for a CreateOrUpdate deployment operation.
// templateFile is a local Azure template.
// See https://github.com/Azure-Samples/resource-manager-go-template-deployment
func Build(mode resources.DeploymentMode, templateFile string, parameters map[string]interface{}) (deployment resources.Deployment, err error) {
	template, err := parseJSONFromFile(templateFile)
	if err != nil {
		return
	}

	finalParameters := map[string]interface{}{}
	for k, v := range parameters {
		addElementToMap(&finalParameters, k, v)
	}

	deployment.Properties = &resources.DeploymentProperties{
		Mode:       mode,
		Template:   template,
		Parameters: &finalParameters,
	}
	return
}

func parseJSONFromFile(filePath string) (*map[string]interface{}, error) {
	text, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	fileMap := map[string]interface{}{}
	if err = json.Unmarshal(text, &fileMap); err != nil {
		return nil, err
	}
	return &fileMap, err
}

func addElementToMap(parameter *map[string]interface{}, key string, value interface{}) {
	(*parameter)[key] = map[string]interface{}{
		"value": value,
	}
}
