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

package kubeletplugin

import (
	"context"
	"net"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2/ktesting"
)

func TestEndpointLifecycle(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tempDir := t.TempDir()
	socketname := "test.sock"
	e := endpoint{dir: tempDir, file: socketname}
	listener, err := e.listen(ctx)
	require.NoError(t, err, "listen")
	assert.FileExists(t, path.Join(tempDir, socketname))
	require.NoError(t, listener.Close(), "close")
	assert.NoFileExists(t, path.Join(tempDir, socketname))
}

func TestEndpointListener(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tempDir := t.TempDir()
	socketname := "test.sock"
	listen := func(ctx2 context.Context, socketpath string) (net.Listener, error) {
		assert.Equal(t, path.Join(tempDir, socketname), socketpath)
		return nil, nil
	}
	e := endpoint{dir: tempDir, file: socketname, listenFunc: listen}
	listener, err := e.listen(ctx)
	require.NoError(t, err, "listen")
	assert.NoFileExists(t, path.Join(tempDir, socketname))
	assert.Nil(t, listener)
}
