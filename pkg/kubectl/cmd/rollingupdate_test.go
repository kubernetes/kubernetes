/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package cmd

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestValidateArgs(t *testing.T) {
	f, _, _ := NewAPIFactory()

	tests := []struct {
		flags     map[string]string
		args      []string
		expectErr bool
		testName  string
	}{
		{
			expectErr: true,
			testName:  "nothing",
		},
		{
			flags:     map[string]string{},
			args:      []string{"foo"},
			expectErr: true,
			testName:  "no file, no image",
		},
		{
			flags: map[string]string{
				"filename": "bar.yaml",
			},
			args:     []string{"foo"},
			testName: "valid file example",
		},
		{
			flags: map[string]string{
				"image": "foo:v2",
			},
			args:     []string{"foo"},
			testName: "missing second image name",
		},
		{
			flags: map[string]string{
				"image": "foo:v2",
			},
			args:     []string{"foo", "foo-v2"},
			testName: "valid image example",
		},
		{
			flags: map[string]string{
				"image":    "foo:v2",
				"filename": "bar.yaml",
			},
			args:      []string{"foo", "foo-v2"},
			expectErr: true,
			testName:  "both filename and image example",
		},
	}
	for _, test := range tests {
		out := &bytes.Buffer{}
		cmd := NewCmdRollingUpdate(f, out)

		if test.flags != nil {
			for key, val := range test.flags {
				cmd.Flags().Set(key, val)
			}
		}
		_, _, _, _, err := validateArguments(cmd, test.args)
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v (%s)", err, test.testName)
		}
		if err == nil && test.expectErr {
			t.Error("unexpected non-error (%s)", test.testName)
		}
	}
}

func TestAddDeploymentHash(t *testing.T) {
	buf := &bytes.Buffer{}
	f, tf, codec := NewAPIFactory()
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "rc"},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}

	podList := &api.PodList{
		Items: []api.Pod{
			{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			{ObjectMeta: api.ObjectMeta{Name: "bar"}},
			{ObjectMeta: api.ObjectMeta{Name: "baz"}},
		},
	}

	seen := util.StringSet{}
	updatedRc := false
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1beta3/namespaces/default/pods" && m == "GET":
				if req.URL.RawQuery != "labelSelector=foo%3Dbar" {
					t.Errorf("Unexpected query string: %s", req.URL.RawQuery)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, podList)}, nil
			case p == "/api/v1beta3/namespaces/default/pods/foo" && m == "PUT":
				seen.Insert("foo")
				obj := readOrDie(t, req, codec)
				podList.Items[0] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[0])}, nil
			case p == "/api/v1beta3/namespaces/default/pods/bar" && m == "PUT":
				seen.Insert("bar")
				obj := readOrDie(t, req, codec)
				podList.Items[1] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[1])}, nil
			case p == "/api/v1beta3/namespaces/default/pods/baz" && m == "PUT":
				seen.Insert("baz")
				obj := readOrDie(t, req, codec)
				podList.Items[2] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[2])}, nil
			case p == "/api/v1beta3/namespaces/default/replicationcontrollers/rc" && m == "PUT":
				updatedRc = true
				return &http.Response{StatusCode: 200, Body: objBody(codec, rc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfig = &client.Config{Version: latest.Version}
	tf.Namespace = "test"

	client, err := f.Client()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.Fail()
		return
	}

	if _, err := addDeploymentKeyToReplicationController(rc, client, "hash", api.NamespaceDefault, buf); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	for _, pod := range podList.Items {
		if !seen.Has(pod.Name) {
			t.Errorf("Missing update for pod: %s", pod.Name)
		}
	}
	if !updatedRc {
		t.Errorf("Failed to update replication controller with new labels")
	}
}

func readOrDie(t *testing.T, req *http.Request, codec runtime.Codec) runtime.Object {
	data, err := ioutil.ReadAll(req.Body)
	if err != nil {
		t.Errorf("Error reading: %v", err)
		t.FailNow()
	}
	obj, err := codec.Decode(data)
	if err != nil {
		t.Errorf("error decoding: %v", err)
		t.FailNow()
	}
	return obj
}
