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

package operationexecutor

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

// fakeRegistrationClient records the call options it was invoked with so
// tests can assert on them, and returns a canned response/error.
type fakeRegistrationClient struct {
	notifyErr  error
	notifyOpts []grpc.CallOption
}

func (f *fakeRegistrationClient) GetInfo(ctx context.Context, in *registerapi.InfoRequest, opts ...grpc.CallOption) (*registerapi.PluginInfo, error) {
	return nil, errors.New("not implemented")
}

func (f *fakeRegistrationClient) NotifyRegistrationStatus(ctx context.Context, in *registerapi.RegistrationStatus, opts ...grpc.CallOption) (*registerapi.RegistrationStatusResponse, error) {
	f.notifyOpts = opts
	if f.notifyErr != nil {
		return nil, f.notifyErr
	}
	return &registerapi.RegistrationStatusResponse{}, nil
}

func hasWaitForReady(opts []grpc.CallOption) bool {
	return slices.Contains(opts, grpc.WaitForReady(true))
}

func TestNotifyPlugin(t *testing.T) {
	tests := []struct {
		name       string
		registered bool
		errStr     string
		notifyErr  error
		wantErr    bool
	}{
		{
			name:       "successful registration notification",
			registered: true,
			errStr:     "",
			wantErr:    false,
		},
		{
			name:       "failed registration is surfaced as an error",
			registered: false,
			errStr:     "registration failed",
			wantErr:    true,
		},
		{
			name:       "transport error from NotifyRegistrationStatus is wrapped",
			registered: true,
			errStr:     "",
			notifyErr:  errors.New("transient failure"),
			wantErr:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := &fakeRegistrationClient{notifyErr: tc.notifyErr}
			og := NewOperationGenerator(nil).(*operationGenerator)

			err := og.notifyPlugin(context.Background(), client, tc.registered, tc.errStr)

			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			// Regardless of outcome, the call must opt in to WaitForReady so
			// that a client that dropped to IDLE/TRANSIENT_FAILURE while the
			// plugin was being registered blocks for reconnect instead of
			// failing fast.
			require.True(t, hasWaitForReady(client.notifyOpts), "expected NotifyRegistrationStatus to be called with grpc.WaitForReady(true), got opts: %#v", client.notifyOpts)
		})
	}
}
