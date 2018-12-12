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

package converter

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

func TestConverter(t *testing.T) {
	sampleObj := `kind: ConversionReview
apiVersion: apiextensions.k8s.io/v1beta1
request:
  uid: 0000-0000-0000-0000
  desiredAPIVersion: stable.example.com/v2
  objects:
    - apiVersion: stable.example.com/v1
      kind: CronTab
      metadata:
        name: my-new-cron-object
      spec:
        cronSpec: "* * * * */5"
        image: my-awesome-cron-image
      hostPort: "localhost:7070"
`
	// First try json, it should fail as the data is taml
	response := httptest.NewRecorder()
	request, err := http.NewRequest("POST", "/convert", strings.NewReader(sampleObj))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Add("Content-Type", "application/json")
	ServeExampleConvert(response, request)
	convertReview := v1beta1.ConversionReview{}
	scheme := runtime.NewScheme()
	jsonSerializer := json.NewSerializer(json.DefaultMetaFactory, scheme, scheme, false)
	if _, _, err := jsonSerializer.Decode(response.Body.Bytes(), nil, &convertReview); err != nil {
		t.Fatal(err)
	}
	if convertReview.Response.Result.Status != v1.StatusFailure {
		t.Fatalf("expected the operation to fail when yaml is provided with json header")
	} else if !strings.Contains(convertReview.Response.Result.Message, "json parse error") {
		t.Fatalf("expected to fail on json parser, but it failed with: %v", convertReview.Response.Result.Message)
	}

	// Now try yaml, and it should successfully convert
	response = httptest.NewRecorder()
	request, err = http.NewRequest("POST", "/convert", strings.NewReader(sampleObj))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Add("Content-Type", "application/yaml")
	ServeExampleConvert(response, request)
	convertReview = v1beta1.ConversionReview{}
	yamlSerializer := json.NewYAMLSerializer(json.DefaultMetaFactory, scheme, scheme)
	if _, _, err := yamlSerializer.Decode(response.Body.Bytes(), nil, &convertReview); err != nil {
		t.Fatalf("cannot decode data: \n %v\n Error: %v", response.Body, err)
	}
	if convertReview.Response.Result.Status != v1.StatusSuccess {
		t.Fatalf("cr conversion failed: %v", convertReview.Response)
	}
	convertedObj := unstructured.Unstructured{}
	if _, _, err := yamlSerializer.Decode(convertReview.Response.ConvertedObjects[0].Raw, nil, &convertedObj); err != nil {
		t.Fatal(err)
	}
	if e, a := "stable.example.com/v2", convertedObj.GetAPIVersion(); e != a {
		t.Errorf("expected= %v, actual= %v", e, a)
	}
	if e, a := "localhost", convertedObj.Object["host"]; e != a {
		t.Errorf("expected= %v, actual= %v", e, a)
	}
	if e, a := "7070", convertedObj.Object["port"]; e != a {
		t.Errorf("expected= %v, actual= %v", e, a)
	}
}
