/*
Copyright The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"mime"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestLoopbackClientProtobuf(t *testing.T) {
	for _, tc := range []struct {
		name           string
		extraFlags     []string
		wantConfigType string
		wantMediaType  string
	}{
		{
			name:           "default enables protobuf",
			wantConfigType: "application/vnd.kubernetes.protobuf",
			wantMediaType:  "application/vnd.kubernetes.protobuf",
		},
		{
			name:           "disabled falls back to json",
			extraFlags:     []string{"--enable-loopback-client-protobuf=false"},
			wantConfigType: "",
			wantMediaType:  "application/json",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			flags := append(framework.DefaultTestServerFlags(), tc.extraFlags...)
			server := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())
			t.Cleanup(server.TearDownFn)

			if got := server.ClientConfig.ContentType; got != tc.wantConfigType {
				t.Errorf("loopback client ContentType = %q, want %q", got, tc.wantConfigType)
			}

			client, err := kubernetes.NewForConfig(server.ClientConfig)
			if err != nil {
				t.Fatal(err)
			}

			result := client.CoreV1().RESTClient().Get().AbsPath("/api/v1/namespaces", metav1.NamespaceDefault).Do(context.Background())
			if err := result.Error(); err != nil {
				t.Fatal(err)
			}
			var contentType string
			result.ContentType(&contentType)
			mediaType, _, err := mime.ParseMediaType(contentType)
			if err != nil {
				t.Fatal(err)
			}
			if mediaType != tc.wantMediaType {
				t.Errorf("response Content-Type = %q, want %q", mediaType, tc.wantMediaType)
			}
		})
	}
}
