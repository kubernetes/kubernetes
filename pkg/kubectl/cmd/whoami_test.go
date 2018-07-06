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

package cmd

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
)

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestCommandWhoAmI(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	whoamiInfo, err := json.Marshal(&user.DefaultInfo{
		Name:   "foo",
		Groups: []string{"bar", "baz"},
	})
	assert.NoError(t, err)
	ns := legacyscheme.Codecs
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Header:     defaultHeader(),
				Body:       ioutil.NopCloser(bytes.NewReader(whoamiInfo)),
			}, nil
		}),
	}
	tf.ClientConfigVal = defaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdWhoAmI(tf, streams)
	cmd.Run(cmd, []string{})
	assert.Equal(t, `User Name: foo
User Groups: bar, baz
`,
		buf.String(), "unexpect output for whoami")
}
