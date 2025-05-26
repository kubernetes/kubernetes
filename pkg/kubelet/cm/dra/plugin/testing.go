/*
Copyright 2025 The Kubernetes Authors.

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

package plugin

import (
	"context"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

// This file contains helper functions intended **solely for use in tests**.
// These APIs are not meant to be used in production code and may change or be
// removed at any time without notice.
//
// While placed in the main package for accessibility by external tests, these
// functions are test-only utilities and should be treated as internal to the
// testing environment.

const (
	// ConnectionTimeout is the default timeout for establishing
	// a gRPC connection to a DRA plugin.
	ConnectionTimeout = 20 * time.Second
	// ConnectionPollInterval is the default interval for polling
	// the connection status of a DRA plugin.
	ConnectionPollInterval = 100 * time.Millisecond
)

// WaitForConnection repeatedly attempts to establish a connection to a DRA
// plugin within the specified timeout. It polls at the given interval until
// a connection is established or the timeout is reached. The function returns
// an error if a connection cannot be established within the timeout period.
func WaitForConnection(ctx context.Context, pluginName, endpoint string, interval, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		p, err := NewDRAPluginClient(pluginName)
		return err == nil && p != nil && p.endpoint == endpoint && p.isConnected(), nil
	})
}
