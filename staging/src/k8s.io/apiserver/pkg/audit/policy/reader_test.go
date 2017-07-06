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

package policy

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/apis/audit"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const policyDef = `
rules:
  - level: None
    nonResourceURLs:
      - /healthz*
      - /version
  - level: RequestResponse
    users: ["tim"]
    userGroups: ["testers", "developers"]
    verbs: ["patch", "delete", "create"]
    resources:
      - group: ""
      - group: "rbac.authorization.k8s.io"
        resources: ["clusterroles", "clusterrolebindings"]
    namespaces: ["default", "kube-system"]
  - level: Metadata
`

var expectedPolicy = &audit.Policy{
	Rules: []audit.PolicyRule{{
		Level:           audit.LevelNone,
		NonResourceURLs: []string{"/healthz*", "/version"},
	}, {
		Level:      audit.LevelRequestResponse,
		Users:      []string{"tim"},
		UserGroups: []string{"testers", "developers"},
		Verbs:      []string{"patch", "delete", "create"},
		Resources: []audit.GroupResources{{}, {
			Group:     "rbac.authorization.k8s.io",
			Resources: []string{"clusterroles", "clusterrolebindings"},
		}},
		Namespaces: []string{"default", "kube-system"},
	}, {
		Level: audit.LevelMetadata,
	}},
}

func TestParser(t *testing.T) {
	// Create a policy file.
	f, err := ioutil.TempFile("", "policy.yaml")
	require.NoError(t, err)
	defer os.Remove(f.Name())

	_, err = f.WriteString(policyDef)
	require.NoError(t, err)
	require.NoError(t, f.Close())

	policy, err := LoadPolicyFromFile(f.Name())
	require.NoError(t, err)

	assert.Len(t, policy.Rules, 3) // Sanity check.
	if !reflect.DeepEqual(policy, expectedPolicy) {
		t.Errorf("Unexpected policy! Diff:\n%s", diff.ObjectDiff(policy, expectedPolicy))
	}
}
