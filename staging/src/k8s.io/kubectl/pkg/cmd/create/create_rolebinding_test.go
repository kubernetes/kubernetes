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

package create

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	rbac "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

var groupVersion = schema.GroupVersion{Group: "rbac.authorization.k8s.io", Version: "v1"}

func TestCreateRoleBinding(t *testing.T) {
	expectBinding := &rbac.RoleBinding{
		TypeMeta: v1.TypeMeta{
			APIVersion: "rbac.authorization.k8s.io/v1",
			Kind:       "RoleBinding",
		},
		ObjectMeta: v1.ObjectMeta{
			Name: "fake-binding",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     "fake-role",
		},
		Subjects: []rbac.Subject{
			{
				Kind:     rbac.UserKind,
				APIGroup: "rbac.authorization.k8s.io",
				Name:     "fake-user",
			},
			{
				Kind:     rbac.GroupKind,
				APIGroup: "rbac.authorization.k8s.io",
				Name:     "fake-group",
			},
			{
				Kind:      rbac.ServiceAccountKind,
				Namespace: "fake-namespace",
				Name:      "fake-account",
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ns := scheme.Codecs

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, groupVersion)
	decoder := ns.DecoderToVersion(info.Serializer, groupVersion)

	tf.Client = &RoleBindingRESTClient{
		RESTClient: &fake.RESTClient{
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/rolebindings" && m == "POST":
					bodyBits, err := ioutil.ReadAll(req.Body)
					if err != nil {
						t.Fatalf("TestCreateRoleBinding error: %v", err)
						return nil, nil
					}

					if obj, _, err := decoder.Decode(bodyBits, nil, &rbac.RoleBinding{}); err == nil {
						if !reflect.DeepEqual(obj.(*rbac.RoleBinding), expectBinding) {
							t.Fatalf("TestCreateRoleBinding: expected:\n%#v\nsaw:\n%#v", expectBinding, obj.(*rbac.RoleBinding))
							return nil, nil
						}
					} else {
						t.Fatalf("TestCreateRoleBinding error, could not decode the request body into rbac.RoleBinding object: %v", err)
						return nil, nil
					}

					responseBinding := &rbac.RoleBinding{}
					responseBinding.Name = "fake-binding"
					return &http.Response{StatusCode: 201, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseBinding))))}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	cmd := NewCmdCreateRoleBinding(tf, genericclioptions.NewTestIOStreamsDiscard())
	cmd.Flags().Set("role", "fake-role")
	cmd.Flags().Set("user", "fake-user")
	cmd.Flags().Set("group", "fake-group")
	cmd.Flags().Set("serviceaccount", "fake-namespace:fake-account")
	cmd.Run(cmd, []string{"fake-binding"})
}

type RoleBindingRESTClient struct {
	*fake.RESTClient
}

func (c *RoleBindingRESTClient) Post() *restclient.Request {
	config := restclient.ContentConfig{
		ContentType:          runtime.ContentTypeJSON,
		GroupVersion:         &groupVersion,
		NegotiatedSerializer: c.NegotiatedSerializer,
	}

	info, _ := runtime.SerializerInfoForMediaType(c.NegotiatedSerializer.SupportedMediaTypes(), runtime.ContentTypeJSON)
	serializers := restclient.Serializers{
		Encoder: c.NegotiatedSerializer.EncoderForVersion(info.Serializer, groupVersion),
		Decoder: c.NegotiatedSerializer.DecoderToVersion(info.Serializer, groupVersion),
	}
	if info.StreamSerializer != nil {
		serializers.StreamingSerializer = info.StreamSerializer.Serializer
		serializers.Framer = info.StreamSerializer.Framer
	}
	return restclient.NewRequest(c, "POST", &url.URL{Host: "localhost"}, c.VersionedAPIPath, config, serializers, nil, nil, 0)
}
