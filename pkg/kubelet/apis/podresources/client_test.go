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

package podresources

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	fakeremote "k8s.io/cri-client/pkg/fake"
)

func TestGetClient(t *testing.T) {
	testCases := map[string]func(string, time.Duration, int) (any, *grpc.ClientConn, error){
		"v1alpha1": func(socket string, timeout time.Duration, maxSize int) (any, *grpc.ClientConn, error) {
			return GetV1alpha1Client(socket, timeout, maxSize)
		},
		"v1": func(socket string, timeout time.Duration, maxSize int) (any, *grpc.ClientConn, error) {
			return GetV1Client(socket, timeout, maxSize)
		},
	}

	for version, getClientFn := range testCases {
		t.Run(version, func(t *testing.T) {
			socketPath, err := fakeremote.GenerateEndpoint()
			require.NoError(t, err)

			client, conn, err := getClientFn(socketPath, 10*time.Second, 1024*1024)
			require.NoError(t, err)
			require.NotNil(t, client)
			require.NoError(t, conn.Close())

			client, conn, err = getClientFn("invalid\x00endpoint", 100*time.Millisecond, 1024*1024)
			require.Error(t, err)
			assert.Nil(t, client)
			assert.Nil(t, conn)
		})
	}
}
