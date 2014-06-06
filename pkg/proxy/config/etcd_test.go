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

package config

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

const TomcatContainerEtcdKey = "/registry/services/tomcat/endpoints/tomcat-3bd5af34"
const TomcatService = "tomcat"
const TomcatContainerId = "tomcat-3bd5af34"

func ValidateJsonParsing(t *testing.T, jsonString string, expectedEndpoints api.Endpoints, expectError bool) {
	endpoints, err := ParseEndpoints(jsonString)
	if err == nil && expectError {
		t.Errorf("ValidateJsonParsing did not get expected error when parsing %s", jsonString)
	}
	if err != nil && !expectError {
		t.Errorf("ValidateJsonParsing got unexpected error %+v when parsing %s", err, jsonString)
	}
	if !reflect.DeepEqual(expectedEndpoints, endpoints) {
		t.Errorf("Didn't get expected endpoints %+v got: %+v", expectedEndpoints, endpoints)
	}
}

func TestParseJsonEndpoints(t *testing.T) {
	ValidateJsonParsing(t, "", api.Endpoints{}, true)
	endpoints := api.Endpoints{
		Name:      "foo",
		Endpoints: []string{"foo", "bar", "baz"},
	}
	data, err := json.Marshal(endpoints)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	ValidateJsonParsing(t, string(data), endpoints, false)
	//	ValidateJsonParsing(t, "[{\"port\":8000,\"name\":\"mysql\",\"machine\":\"foo\"},{\"port\":9000,\"name\":\"mysql\",\"machine\":\"bar\"}]", []string{"foo:8000", "bar:9000"}, false)
}
