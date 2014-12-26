/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubecfg

import (
	"encoding/json"
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func ValidateObject(data []byte, c *client.Client) error {
	var obj interface{}
	err := json.Unmarshal(data, &obj)
	if err != nil {
		return err
	}
	apiVersion, found := obj.(map[string]interface{})["apiVersion"]
	if !found {
		return fmt.Errorf("couldn't find apiVersion in object")
	}

	schemaData, err := c.RESTClient.Get().
		AbsPath("/swaggerapi/api").
		Prefix(apiVersion.(string)).
		Do().
		Raw()
	if err != nil {
		return err
	}
	schema, err := validation.NewSwaggerSchemaFromBytes(schemaData)
	if err != nil {
		return err
	}
	return schema.ValidateBytes(data)
}
