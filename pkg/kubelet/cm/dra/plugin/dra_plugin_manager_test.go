/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"math/rand/v2"
	"path"
	goruntime "runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	drahealthv1 "k8s.io/kubelet/pkg/apis/dra-health/v1"
	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPickHealthService(t *testing.T) {
	for name, tc := range map[string]struct {
		supportedServices []string
		want              string
	}{
		"none": {
			supportedServices: []string{"v1beta1.DRAPlugin"},
			want:              "",
		},
		"empty": {
			supportedServices: nil,
			want:              "",
		},
		// Drivers which shipped before v1 existed only serve v1alpha1.
		// The kubelet keeps consuming it for three releases of transition,
		// see healthServicesSupportedByKubelet.
		"only-v1alpha1": {
			supportedServices: []string{"v1beta1.DRAPlugin", drahealthv1alpha1.DRAResourceHealthService},
			want:              drahealthv1alpha1.DRAResourceHealthService,
		},
		"only-v1": {
			supportedServices: []string{"v1beta1.DRAPlugin", drahealthv1.DRAResourceHealthService},
			want:              drahealthv1.DRAResourceHealthService,
		},
		"both-picks-v1": {
			supportedServices: []string{drahealthv1alpha1.DRAResourceHealthService, drahealthv1.DRAResourceHealthService},
			want:              drahealthv1.DRAResourceHealthService,
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := pickHealthService(tc.supportedServices); got != tc.want {
				t.Errorf("pickHealthService(%v) = %q, want %q", tc.supportedServices, got, tc.want)
			}
		})
	}
}

func TestAddSameName(t *testing.T) {
	tCtx := ktesting.Init(t)
	// name will have a random value to avoid conflicts
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))

	// ensure the plugin we are using is registered
	tmp := t.TempDir()
	oldSock := path.Join(tmp, "old.sock")
	newSock := path.Join(tmp, "new.sock")
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, nil, 0, tmp)
	tCtx.ExpectNoError(draPlugins.add(driverName, oldSock, "", "", defaultClientCallTimeout), "add first plugin")
	p, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get first plugin")

	// Same name, same endpoint -> error.
	require.Error(tCtx, draPlugins.add(driverName, oldSock, "", "", defaultClientCallTimeout))

	tCtx.ExpectNoError(draPlugins.add(driverName, newSock, "", "", defaultClientCallTimeout), "add second plugin")
	p2, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get second plugin")
	if p == p2 {
		tCtx.Fatal("expected to get second plugin, got first one again")
	}

	// Remove old plugin.
	draPlugins.remove(p.driverName, p.endpoint)
	plugin, err := draPlugins.GetPlugin(driverName)

	// Now we should have p2 left.
	tCtx.ExpectNoError(err, "get plugin")
	if p2 != plugin {
		tCtx.Fatal("expected to get second plugin again, got something else")
	}
}

// TestRegisterPluginRejectsUntrustedInput ensures RegisterPlugin/ValidatePlugin
// rejects invalid driver name or endpoint. The gRPC GetInfo response the values
// come from is untrusted, so kubelet must not dial arbitrary UNIX socket paths
// or let one plugin hijack another driver's registration by advertising its name.
func TestRegisterPluginRejectsUntrustedInput(t *testing.T) {
	if goruntime.GOOS == "windows" {
		// This test uses hardcoded POSIX-style paths ("/var/lib/kubelet/...")
		// which filepath.IsAbs does not treat as absolute on Windows. The
		// validation code itself is portable; only the fixtures aren't.
		t.Skip("test fixtures use POSIX-style paths")
	}
	services := []string{drapb.DRAPluginService}
	longName := strings.Repeat("a", 64)
	longEndpoint := "/var/lib/kubelet/plugins/example.com/very_long_endpoint_name_that_exceeds_the_unix_socket_limit_108_bytes.sock"
	rootDir := "/var/lib/kubelet"

	for _, tt := range []struct {
		name       string
		driverName string
		endpoint   string
		errFrag    string
	}{
		{"empty-driver", "", "/var/lib/kubelet/plugins/foo/plugin.sock", "Required value"},
		{"driver-with-slash", "../evil", "/var/lib/kubelet/plugins/foo/plugin.sock", "invalid DRA driver name"},
		{"driver-with-space", "evil driver", "/var/lib/kubelet/plugins/foo/plugin.sock", "invalid DRA driver name"},
		{"driver-too-long", longName, "/var/lib/kubelet/plugins/foo/plugin.sock", "Too long"},
		{"empty-endpoint", "good.example.com", "", "empty DRA plugin endpoint"},
		{"relative-endpoint", "good.example.com", "plugin.sock", "must be an absolute path"},
		{"endpoint-too-long", "good.example.com", longEndpoint, "longer than"},
		{"endpoint-outside-rootDir", "good.example.com", "/run/some-other/socket.sock", "is not inside"},
		{"endpoint-dotdot-escapes-rootDir", "good.example.com", "/var/lib/kubelet/../../etc/passwd", "is not inside"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			draPlugins := NewDRAPluginManager(tCtx, nil, nil, nil, 0, rootDir)
			t.Cleanup(draPlugins.Stop)

			err := draPlugins.RegisterPlugin(tCtx, tt.driverName, tt.endpoint, services, nil)
			require.Error(t, err, "RegisterPlugin must reject untrusted input")
			assert.Contains(t, err.Error(), tt.errFrag)

			err = draPlugins.ValidatePlugin(tCtx, tt.driverName, tt.endpoint, services)
			require.Error(t, err, "ValidatePlugin must reject untrusted input")
			assert.Contains(t, err.Error(), tt.errFrag)
		})
	}
}

// TestRegisterPluginAcceptsEndpointInsideRootDir sanity-checks that the
// containment rule is lexical, not literal: both plugins/ and
// plugins_registry/ subtrees are accepted (real DRA drivers use each), and
// dotdot components that Clean() reduces to a path inside rootDir don't
// cause a false rejection.
func TestRegisterPluginAcceptsEndpointInsideRootDir(t *testing.T) {
	if goruntime.GOOS == "windows" {
		// This test uses hardcoded POSIX-style paths ("/var/lib/kubelet/...")
		// which filepath.IsAbs does not treat as absolute on Windows. The
		// validation code itself is portable; only the fixtures aren't.
		t.Skip("test fixtures use POSIX-style paths")
	}
	services := []string{drapb.DRAPluginService}
	rootDir := "/var/lib/kubelet"

	for _, endpoint := range []string{
		"/var/lib/kubelet/plugins/example.com/dra.sock",
		"/var/lib/kubelet/plugins/example.com/rolling/dra.sock",
		"/var/lib/kubelet/plugins_registry/example.com-common.sock", // "1 common socket" layout
		"/var/lib/kubelet/plugins/./example.com/../dra.sock",        // Clean() reduces to inside
	} {
		t.Run(endpoint, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			draPlugins := NewDRAPluginManager(tCtx, nil, nil, nil, 0, rootDir)
			t.Cleanup(draPlugins.Stop)

			// We don't spin up a gRPC server here — only checking that the
			// validation step passes. ValidatePlugin does not open a
			// connection, so it fully exercises the validation.
			require.NoError(t, draPlugins.ValidatePlugin(tCtx, "example.com", endpoint, services))
		})
	}
}

func TestDelete(t *testing.T) {
	tCtx := ktesting.Init(t)
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))
	tmp := t.TempDir()
	socketFile := path.Join(tmp, "dra.sock")

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, &mockStreamHandler{}, 0, tmp)
	tCtx.ExpectNoError(draPlugins.add(driverName, socketFile, "", "", defaultClientCallTimeout), "add plugin")

	draPlugins.remove(driverName, socketFile)

	_, err := draPlugins.GetPlugin(driverName)
	require.Error(t, err, "plugin should not exist after being removed")
}
