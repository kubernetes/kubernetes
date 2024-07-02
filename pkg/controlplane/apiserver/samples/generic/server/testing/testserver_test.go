/*
Copyright 2023 The Kubernetes Authors.

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

package testing

import (
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/test/integration/framework"
)

func TestTestServer(t *testing.T) {
	var tearDownFn TearDownFunc
	defer func() {
		if tearDownFn != nil {
			tearDownFn()
		}
	}()

	etcdConfig := framework.SharedEtcd()
	server := StartTestServerOrDie(t, nil, nil, etcdConfig)
	tearDownFn = server.TearDownFn

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	groups, err := client.Discovery().ServerPreferredResources()
	require.NoError(t, err)

	for _, g := range groups {
		for _, r := range g.APIResources {
			t.Logf("Found resource %s.%s", r.Name, g.GroupVersion)
		}
	}
}
