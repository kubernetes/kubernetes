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
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

func TestConverterYAML(t *testing.T) {
	cases := []struct {
		apiVersion     string
		contentType    string
		expected400Err string
	}{
		{
			apiVersion:     "apiextensions.k8s.io/v1beta1",
			contentType:    "application/json",
			expected400Err: "json parse error",
		},
		{
			apiVersion:  "apiextensions.k8s.io/v1beta1",
			contentType: "application/yaml",
		},
		{
			apiVersion:     "apiextensions.k8s.io/v1",
			contentType:    "application/json",
			expected400Err: "json parse error",
		},
		{
			apiVersion:  "apiextensions.k8s.io/v1",
			contentType: "application/yaml",
		},
	}
	sampleObjTemplate := `kind: ConversionReview
apiVersion: %s
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
	for _, tc := range cases {
		t.Run(tc.apiVersion+" "+tc.contentType, func(t *testing.T) {
			sampleObj := fmt.Sprintf(sampleObjTemplate, tc.apiVersion)
			// First try json, it should fail as the data is taml
			response := httptest.NewRecorder()
			request, err := http.NewRequest("POST", "/convert", strings.NewReader(sampleObj))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Add("Content-Type", tc.contentType)
			ServeExampleConvert(response, request)
			convertReview := apiextensionsv1.ConversionReview{}
			scheme := runtime.NewScheme()
			if len(tc.expected400Err) > 0 {
				body := response.Body.Bytes()
				if !bytes.Contains(body, []byte(tc.expected400Err)) {
					t.Fatalf("expected to fail on '%s', but it failed with: %s", tc.expected400Err, string(body))
				}
				return
			}

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
		})
	}
}
