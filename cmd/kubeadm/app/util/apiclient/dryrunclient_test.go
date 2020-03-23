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

package apiclient

import (
	"bytes"
	"testing"

	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	core "k8s.io/client-go/testing"
)

func TestLogDryRunAction(t *testing.T) {
	var tests = []struct {
		name          string
		action        core.Action
		expectedBytes []byte
		buf           *bytes.Buffer
	}{
		{
			name:   "action GET on services",
			action: core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "services"}, "default", "kubernetes"),
			expectedBytes: []byte(`[dryrun] Would perform action GET on resource "services" in API group "core/v1"
[dryrun] Resource name: "kubernetes"
`),
		},
		{
			name:   "action GET on clusterrolebindings",
			action: core.NewRootGetAction(schema.GroupVersionResource{Group: rbac.GroupName, Version: rbac.SchemeGroupVersion.Version, Resource: "clusterrolebindings"}, "system:node"),
			expectedBytes: []byte(`[dryrun] Would perform action GET on resource "clusterrolebindings" in API group "rbac.authorization.k8s.io/v1"
[dryrun] Resource name: "system:node"
`),
		},
		{
			name:   "action LIST on services",
			action: core.NewListAction(schema.GroupVersionResource{Version: "v1", Resource: "services"}, schema.GroupVersionKind{Version: "v1", Kind: "Service"}, "default", metav1.ListOptions{}),
			expectedBytes: []byte(`[dryrun] Would perform action LIST on resource "services" in API group "core/v1"
`),
		},
		{
			name: "action CREATE on services",
			action: core.NewCreateAction(schema.GroupVersionResource{Version: "v1", Resource: "services"}, "default", &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: v1.ServiceSpec{
					ClusterIP: "1.1.1.1",
				},
			}),
			expectedBytes: []byte(`[dryrun] Would perform action CREATE on resource "services" in API group "core/v1"
	apiVersion: v1
	kind: Service
	metadata:
	  creationTimestamp: null
	  name: foo
	spec:
	  clusterIP: 1.1.1.1
	status:
	  loadBalancer: {}
`),
		},
		{
			name:   "action PATCH on nodes",
			action: core.NewPatchAction(schema.GroupVersionResource{Version: "v1", Resource: "nodes"}, "default", "my-node", "application/strategic-merge-patch+json", []byte(`{"spec":{"taints":[{"key": "foo", "value": "bar"}]}}`)),
			expectedBytes: []byte(`[dryrun] Would perform action PATCH on resource "nodes" in API group "core/v1"
[dryrun] Resource name: "my-node"
[dryrun] Attached patch:
	{"spec":{"taints":[{"key": "foo", "value": "bar"}]}}
`),
		},
		{
			name:   "action DELETE on pods",
			action: core.NewDeleteAction(schema.GroupVersionResource{Version: "v1", Resource: "pods"}, "default", "my-pod"),
			expectedBytes: []byte(`[dryrun] Would perform action DELETE on resource "pods" in API group "core/v1"
[dryrun] Resource name: "my-pod"
`),
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			rt.buf = bytes.NewBufferString("")
			logDryRunAction(rt.action, rt.buf, DefaultMarshalFunc)
			actualBytes := rt.buf.Bytes()

			if !bytes.Equal(actualBytes, rt.expectedBytes) {
				t.Errorf(
					"failed LogDryRunAction:\n\texpected bytes: %q\n\t  actual: %q",
					rt.expectedBytes,
					actualBytes,
				)
			}
		})
	}
}
