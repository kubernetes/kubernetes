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

package cmd

import (
	"bytes"
	"io"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type testClusterRolePrinter struct {
	CachedClusterRole *rbac.ClusterRole
}

func (t *testClusterRolePrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.CachedClusterRole = obj.(*rbac.ClusterRole)
	return nil
}

func (t *testClusterRolePrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *testClusterRolePrinter) HandledResources() []string {
	return []string{}
}

func TestCreateClusterRole(t *testing.T) {
	clusterRoleName := "my-cluster-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	printer := &testClusterRolePrinter{}
	tf.Printer = printer
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	tests := map[string]struct {
		verbs               string
		resources           string
		resourceNames       string
		expectedClusterRole *rbac.ClusterRole
	}{
		"test-duplicate-resources": {
			verbs:     "get,watch,list",
			resources: "pods,pods",
			expectedClusterRole: &rbac.ClusterRole{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"pods"},
						APIGroups:     []string{""},
						ResourceNames: []string{},
					},
				},
			},
		},
		"test-valid-case-with-multiple-apigroups": {
			verbs:     "get,watch,list",
			resources: "pods,deployments.extensions",
			expectedClusterRole: &rbac.ClusterRole{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"pods"},
						APIGroups:     []string{""},
						ResourceNames: []string{},
					},
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"deployments"},
						APIGroups:     []string{"extensions"},
						ResourceNames: []string{},
					},
				},
			},
		},
	}

	for name, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreateClusterRole(f, buf)
		cmd.Flags().Set("dry-run", "true")
		cmd.Flags().Set("output", "object")
		cmd.Flags().Set("verb", test.verbs)
		cmd.Flags().Set("resource", test.resources)
		if test.resourceNames != "" {
			cmd.Flags().Set("resource-name", test.resourceNames)
		}
		cmd.Run(cmd, []string{clusterRoleName})
		if !reflect.DeepEqual(test.expectedClusterRole, printer.CachedClusterRole) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expectedClusterRole, printer.CachedClusterRole)
		}
	}
}
