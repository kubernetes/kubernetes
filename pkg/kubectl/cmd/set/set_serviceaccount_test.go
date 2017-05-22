/*
Copyright YEAR The Kubernetes Authors.

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

package set

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"

	"github.com/stretchr/testify/assert"
)

func TestServiceAccountLocal(t *testing.T) {
	inputs := []struct {
		yaml     string
		apiGroup string
	}{
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/replication.yaml", apiGroup: api.GroupName},
		{yaml: "../../../../test/fixtures/doc-yaml/admin/daemon.yaml", apiGroup: extensions.GroupName},
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/replicaset/redis-slave.yaml", apiGroup: extensions.GroupName},
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/job.yaml", apiGroup: batch.GroupName},
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/deployment.yaml", apiGroup: extensions.GroupName},
		{yaml: "../../../../examples/storage/cassandra/cassandra-statefulset.yaml", apiGroup: apps.GroupName},
	}

	f, tf, _, ns := cmdtesting.NewAPIFactory()
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}

	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(api.GroupName).GroupVersion}}

	out := bytes.NewBuffer([]byte{})
	cmd := NewCmdServiceAccount(f, out, out)
	cmd.SetOutput(out)
	cmd.Flags().Set("output", "yaml")
	cmd.Flags().Set("local", "true")
	for _, input := range inputs {
		testapi.Default = testapi.Groups[input.apiGroup]
		tf.Printer = printers.NewVersionedPrinter(&printers.YAMLPrinter{}, testapi.Default.Converter(), *testapi.Default.GroupVersion())
		saConfig := ServiceAccountConfig{fileNameOptions: resource.FilenameOptions{
			Filenames: []string{input.yaml}},
			Out:   out,
			Local: true}
		err := saConfig.Complete(f, cmd, []string{"serviceaccount1"})
		if err == nil {
			err = saConfig.Validate()
		}
		if err == nil {
			err = saConfig.Run()
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assert.True(t, strings.Contains(out.String(), "serviceAccountName: serviceaccount1"))
	}
}
