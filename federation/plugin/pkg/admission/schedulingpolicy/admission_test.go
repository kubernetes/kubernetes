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

package schedulingpolicy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

func TestNewAdmissionController(t *testing.T) {
	tempfile, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("Unexpected error while creating temporary file: %v", err)
	}
	p := tempfile.Name()
	defer os.Remove(p)

	kubeconfig := `
clusters:
  - name: foo
    cluster:
      server: https://example.com
users:
  - name: alice
    user:
      token: deadbeef
contexts:
  - name: default
    context:
      cluster: foo
      user: alice
current-context: default
`

	if _, err := tempfile.WriteString(kubeconfig); err != nil {
		t.Fatalf("Unexpected error while writing test kubeconfig file: %v", err)
	}

	tests := []struct {
		note    string
		input   string
		wantErr bool
	}{
		{"no config", "", true},
		{"bad json", `{"foo": `, true},
		{"bad yaml", `{foo" `, true},
		{
			"missing kubeconfig",
			`{"foo": {}}`,
			true,
		},
		{
			"kubeconfig not found",
			`{
				"kubeconfig": "/kube-federation-scheduling-policy-file-not-found-test"
			}`,
			true,
		},
		{
			"bad retry backoff",
			fmt.Sprintf(`
				{
					"kubeconfig": %q,
					"retryBackoff": -1
				}
				`, p),
			true,
		},
		{
			"a valid config",
			fmt.Sprintf(`
				{
					"kubeconfig": %q
				}
				`, p),
			false,
		},
		{
			"a valid config with retry backoff",
			fmt.Sprintf(`
				{
					"kubeconfig": %q,
					"retryBackoff": 200
				}
				`, p),
			false,
		},
	}

	for _, tc := range tests {
		var file io.Reader
		if tc.input == "" {
			file = nil
		} else {
			file = bytes.NewBufferString(tc.input)
		}

		_, err := newAdmissionController(file)

		if tc.wantErr && err == nil {
			t.Errorf("%v: Expected error", tc.note)
		} else if !tc.wantErr && err != nil {
			t.Errorf("%v: Unexpected error: %v", tc.note, err)
		}
	}
}

func TestAdmitQueryPayload(t *testing.T) {
	var body interface{}

	serve := func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("Unexpected error reading admission payload: %v", err)
		}

		// No errors or annotations.
		w.Write([]byte(`{}`))
	}

	controller, err := newControllerWithTestServer(serve, true)
	if err != nil {
		t.Fatalf("Unexpected error while creating test admission controller/server: %v", err)
	}

	rs := makeReplicaSet()
	rs.Spec.MinReadySeconds = 100
	attrs := makeAdmissionRecord(rs)
	err = controller.Admit(attrs)

	if err != nil {
		t.Fatalf("Unexpected error from admission controller: %v", err)
	}

	obj := body.(map[string]interface{})
	metadata := obj["metadata"].(map[string]interface{})
	spec := obj["spec"].(map[string]interface{})
	name := metadata["name"].(string)
	minReadySeconds := spec["minReadySeconds"].(float64)

	expectedName := "myapp"
	if name != expectedName {
		t.Fatalf("Expected replicaset.metadata.name to be %v but got: %v", expectedName, name)
	}

	expectedMinReadySeconds := float64(100)
	if minReadySeconds != expectedMinReadySeconds {
		t.Fatalf("Expected replicaset.spec.minReadySeconds to be %v but got: %v", expectedMinReadySeconds, minReadySeconds)
	}
}

func TestAdmitFailInternal(t *testing.T) {
	serve := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}

	controller, err := newControllerWithTestServer(serve, false)
	if err != nil {
		t.Fatalf("Unexpected error while creating test admission controller/server: %v", err)
	}

	mockClient := &fake.Clientset{}
	mockClient.AddReactor("list", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("unknown error")
	})

	controller.SetInternalKubeClientSet(mockClient)

	attrs := makeAdmissionRecord(makeReplicaSet())
	err = controller.Admit(attrs)

	if err == nil {
		t.Fatalf("Expected admission controller to fail closed")
	}
}

func TestAdmitPolicyDoesNotExist(t *testing.T) {

	controller, err := newControllerWithTestServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
	}, false)
	if err != nil {
		t.Fatalf("Unexpected error while creating test admission controller/server: %v", err)
	}

	attrs := makeAdmissionRecord(makeReplicaSet())
	err = controller.Admit(attrs)

	if err != nil {
		t.Fatalf("Expected admission controller to fail open but got error: %v", err)
	}
}

func TestAdmitFailClosed(t *testing.T) {
	tests := []struct {
		note       string
		statusCode int
		body       string
	}{
		{"server error", 500, ""},
		{"unmarshal error", 200, "{"},
		{"undefined result", 404, ``},
		{"policy errors", 200, `{"errors": ["conflicting replica-set-preferences"]}`},
	}

	for _, tc := range tests {

		serve := func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tc.statusCode)
			if len(tc.body) > 0 {
				w.Write([]byte(tc.body))
			}
		}

		controller, err := newControllerWithTestServer(serve, true)

		if err != nil {
			t.Errorf("%v: Unexpected error while creating test admission controller/server: %v", tc.note, err)
			continue
		}

		obj := makeReplicaSet()
		attrs := admission.NewAttributesRecord(obj, nil, obj.GroupVersionKind(), obj.Namespace, obj.Name, api.Resource("replicasets").WithVersion("version"), "", admission.Create, &user.DefaultInfo{})
		err = controller.Admit(attrs)
		if err == nil {
			t.Errorf("%v: Expected admission controller to fail closed", tc.note)
		}

	}
}

func TestAdmitRetries(t *testing.T) {
	var numQueries int

	controller, err := newControllerWithTestServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		numQueries++
	}, true)

	if err != nil {
		t.Fatalf("Unexpected error while creating test admission controller/server: %v", err)
	}

	err = controller.Admit(makeAdmissionRecord(makeReplicaSet()))

	if err == nil {
		t.Fatalf("Expected admission controller to fail closed")
	}

	if numQueries <= 1 {
		t.Fatalf("Expected multiple queries/retries but got (numQueries): %v", numQueries)
	}
}

func TestAdmitSuccessWithAnnotationMerge(t *testing.T) {
	controller, err := newControllerWithTestServer(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`
		{
			"annotations": {
				"foo": "bar-2"
			}
		}
		`))
	}, true)
	if err != nil {
		t.Fatalf("Unexpected error while creating test admission controller/server: %v", err)
	}

	obj := makeReplicaSet()
	obj.Annotations = map[string]string{}
	obj.Annotations["foo"] = "bar"
	obj.Annotations["bar"] = "baz"

	attrs := admission.NewAttributesRecord(obj, nil, obj.GroupVersionKind(), obj.Namespace, obj.Name, api.Resource("replicasets").WithVersion("version"), "", admission.Create, &user.DefaultInfo{})
	err = controller.Admit(attrs)
	if err != nil {
		t.Fatalf("Unexpected error from admission controller: %v", err)
	}

	annotations := attrs.GetObject().(*extensionsv1.ReplicaSet).Annotations
	expected := map[string]string{
		"foo": "bar-2",
		"bar": "baz",
	}

	if !reflect.DeepEqual(annotations, expected) {
		t.Fatalf("Expected annotations to be %v but got: %v", expected, annotations)
	}
}

func newControllerWithTestServer(f func(w http.ResponseWriter, r *http.Request), policiesExist bool) (*admissionController, error) {
	server, err := newTestServer(f)
	if err != nil {
		return nil, err
	}

	kubeConfigFile, err := makeKubeConfigFile(server.URL, "/some/path/to/decision")
	if err != nil {
		return nil, err
	}

	defer os.Remove(kubeConfigFile)

	configFile, err := makeAdmissionControlConfigFile(kubeConfigFile)
	if err != nil {
		return nil, err
	}

	defer os.Remove(configFile)

	file, err := os.Open(configFile)
	if err != nil {
		return nil, err
	}

	controller, err := newAdmissionController(file)
	if err != nil {
		return nil, err
	}

	mockClient := &fake.Clientset{}

	var items []api.ConfigMap

	if policiesExist {
		items = append(items, api.ConfigMap{})
	}

	mockClient.AddReactor("list", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		if action.GetNamespace() == policyConfigMapNamespace {
			return true, &api.ConfigMapList{Items: items}, nil
		}
		return true, nil, nil
	})

	controller.SetInternalKubeClientSet(mockClient)

	return controller, nil
}

func newTestServer(f func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, error) {
	server := httptest.NewUnstartedServer(http.HandlerFunc(f))
	server.Start()
	return server, nil
}

func makeAdmissionControlConfigFile(kubeConfigFile string) (string, error) {
	tempfile, err := ioutil.TempFile("", "")
	if err != nil {
		return "", err
	}

	p := tempfile.Name()

	configFileTmpl := `
kubeconfig: {{ .KubeConfigFile }}
retryBackoff: {{ .RetryBackoff }}
`
	type configFileTemplateInput struct {
		KubeConfigFile string
		RetryBackoff   int
	}

	input := configFileTemplateInput{
		KubeConfigFile: kubeConfigFile,
		RetryBackoff:   1,
	}

	tmpl, err := template.New("scheduling-policy-config").Parse(configFileTmpl)
	if err != nil {
		return "", err
	}

	if err := tmpl.Execute(tempfile, input); err != nil {
		return "", err
	}

	return p, nil
}

func makeKubeConfigFile(baseURL, path string) (string, error) {
	tempfile, err := ioutil.TempFile("", "")
	if err != nil {
		return "", err
	}

	p := tempfile.Name()

	kubeConfigTmpl := `
clusters:
  - name: test
    cluster:
      server: {{ .BaseURL }}{{ .Path }}
users:
  - name: alice
    user:
      token: deadbeef
contexts:
  - name: default
    context:
      cluster: test
      user: alice
current-context: default`

	type kubeConfigTemplateInput struct {
		BaseURL string
		Path    string
	}

	input := kubeConfigTemplateInput{
		BaseURL: baseURL,
		Path:    path,
	}

	tmpl, err := template.New("kubeconfig").Parse(kubeConfigTmpl)
	if err != nil {
		return "", err
	}

	if err := tmpl.Execute(tempfile, input); err != nil {
		return "", err
	}

	return p, nil
}

func makeAdmissionRecord(obj *extensionsv1.ReplicaSet) admission.Attributes {
	return admission.NewAttributesRecord(obj, nil, obj.GroupVersionKind(), obj.Namespace, obj.Name, api.Resource("replicasets").WithVersion("version"), "", admission.Create, &user.DefaultInfo{})
}

func makeReplicaSet() *extensionsv1.ReplicaSet {
	return &extensionsv1.ReplicaSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicaSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "myapp",
		},
		Spec: extensionsv1.ReplicaSetSpec{},
	}
}
