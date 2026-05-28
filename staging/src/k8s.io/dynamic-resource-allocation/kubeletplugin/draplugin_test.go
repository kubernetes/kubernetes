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

package kubeletplugin

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	drahealthv1 "k8s.io/kubelet/pkg/apis/dra-health/v1"
)

// stubPlugin implements the minimal [DRAPlugin] interface and opts out of
// health reporting by returning ErrHealthNotSupported.
type stubPlugin struct{}

func (s *stubPlugin) PrepareResourceClaims(ctx context.Context, claims []*resourceapi.ResourceClaim) (map[types.UID]PrepareResult, error) {
	return nil, nil
}

func (s *stubPlugin) UnprepareResourceClaims(ctx context.Context, claims []NamespacedObject) (map[types.UID]error, error) {
	return nil, nil
}

func (s *stubPlugin) HandleError(ctx context.Context, err error, msg string) {}

func (s *stubPlugin) WatchHealthStatus(ctx context.Context, reports chan<- DeviceHealthReport) error {
	return ErrHealthNotSupported
}

// healthPlugin is a [DRAPlugin] which supports health reporting by sending a
// fixed list of reports and then returning.
type healthPlugin struct {
	stubPlugin
	reports []DeviceHealthReport
}

func (p *healthPlugin) WatchHealthStatus(ctx context.Context, reports chan<- DeviceHealthReport) error {
	for _, r := range p.reports {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case reports <- r:
		}
	}
	return nil
}

// TestStartWithoutHealth ensures that a driver which opts out of health
// reporting starts cleanly.
func TestStartWithoutHealth(t *testing.T) {
	ctx := t.Context()
	helper, err := Start(ctx, &stubPlugin{},
		DriverName("test-driver"),
		KubeClient(fake.NewClientset()),
		// Don't start any sockets, this test only cares about the wiring.
		RegistrationService(false),
		DRAService(false),
	)
	require.NoError(t, err)
	helper.Stop()
}

// TestStartWithoutDRAAPI ensures that disabling all DRA gRPC API versions
// fails fast at Start instead of registering a driver which only advertises
// the health add-on and which the kubelet then cannot use.
func TestStartWithoutDRAAPI(t *testing.T) {
	ctx := t.Context()
	_, err := Start(ctx, &stubPlugin{},
		DriverName("test-driver"),
		KubeClient(fake.NewClientset()),
		RegistrationService(false),
		DRAService(false),
		NodeV1(false),
		NodeV1beta1(false),
	)
	require.ErrorContains(t, err, "no supported DRA gRPC API")
}

// fakeHealthStream captures the responses sent by the helper's gRPC bridge.
type fakeHealthStream struct {
	grpc.ServerStream
	ctx  context.Context
	sent []*drahealthv1.NodeWatchResourcesResponse
}

func (f *fakeHealthStream) Context() context.Context { return f.ctx }

func (f *fakeHealthStream) Send(resp *drahealthv1.NodeWatchResourcesResponse) error {
	f.sent = append(f.sent, resp)
	return nil
}

// TestHealthServerBridge verifies that the helper's gRPC bridge forwards the
// version-neutral reports produced by [DRAPlugin.WatchHealthStatus] and converts
// them to the v1 wire types.
func TestHealthServerBridge(t *testing.T) {
	plugin := &healthPlugin{
		reports: []DeviceHealthReport{
			{Devices: []DeviceHealth{
				{PoolName: "pool", DeviceName: "dev0", Health: HealthStatusHealthy, Message: "ok"},
				{PoolName: "pool", DeviceName: "dev1", Health: HealthStatusUnhealthy},
			}},
		},
	}
	stream := &fakeHealthStream{ctx: t.Context()}

	bridge := &healthServerBridge{plugin: plugin}
	err := bridge.NodeWatchResources(&drahealthv1.NodeWatchResourcesRequest{}, stream)
	require.NoError(t, err)

	require.Len(t, stream.sent, 1)
	devices := stream.sent[0].GetDevices()
	require.Len(t, devices, 2)

	require.Equal(t, "pool", devices[0].GetDevice().GetPoolName())
	require.Equal(t, "dev0", devices[0].GetDevice().GetDeviceName())
	require.Equal(t, drahealthv1.HealthStatus_HEALTHY, devices[0].GetHealth())
	require.Equal(t, "ok", devices[0].GetMessage())

	require.Equal(t, "dev1", devices[1].GetDevice().GetDeviceName())
	require.Equal(t, drahealthv1.HealthStatus_UNHEALTHY, devices[1].GetHealth())
}

// stalePlugin sends one report and then goes silent, simulating a driver
// whose health monitoring wedged. HandleError calls are recorded.
type stalePlugin struct {
	stubPlugin
	report    DeviceHealthReport
	handleErr chan error
}

func (p *stalePlugin) HandleError(ctx context.Context, err error, msg string) {
	select {
	case p.handleErr <- err:
	default:
	}
}

func (p *stalePlugin) WatchHealthStatus(ctx context.Context, reports chan<- DeviceHealthReport) error {
	select {
	case <-ctx.Done():
		return nil
	case reports <- p.report:
	}
	<-ctx.Done()
	return nil
}

// TestHealthServerBridgeStaleWatchdog verifies that the helper surfaces a
// recoverable error through HandleError when the driver stops re-sending
// health reports within the health check timeout, without re-sending on the
// driver's behalf.
func TestHealthServerBridgeStaleWatchdog(t *testing.T) {
	origInterval := healthStaleCheckInterval
	healthStaleCheckInterval = 10 * time.Millisecond
	defer func() { healthStaleCheckInterval = origInterval }()

	plugin := &stalePlugin{
		report: DeviceHealthReport{Devices: []DeviceHealth{
			{PoolName: "pool", DeviceName: "dev0", Health: HealthStatusHealthy, HealthCheckTimeout: 20 * time.Millisecond},
		}},
		handleErr: make(chan error, 1),
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	stream := &fakeHealthStream{ctx: ctx}

	bridge := &healthServerBridge{plugin: plugin}
	done := make(chan error, 1)
	go func() {
		done <- bridge.NodeWatchResources(&drahealthv1.NodeWatchResourcesRequest{}, stream)
	}()

	select {
	case err := <-plugin.handleErr:
		require.ErrorIs(t, err, ErrRecoverable)
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for the stale-health watchdog to fire")
	}

	// The helper must not have re-sent anything: only the driver's single
	// report reached the kubelet.
	cancel()
	require.NoError(t, <-done)
	require.Len(t, stream.sent, 1)
}

// TestHealthServerBridgeNotSupported verifies that a driver which does not
// support health reporting fails the kubelet's health stream with
// Unimplemented, which tells the kubelet to stop watching.
func TestHealthServerBridgeNotSupported(t *testing.T) {
	stream := &fakeHealthStream{ctx: t.Context()}

	bridge := &healthServerBridge{plugin: &stubPlugin{}}
	err := bridge.NodeWatchResources(&drahealthv1.NodeWatchResourcesRequest{}, stream)
	require.Equal(t, codes.Unimplemented, status.Code(err))
	require.Empty(t, stream.sent)
}

// TestMinHealthCheckTimeout verifies that the stale-health watchdog uses the
// tightest lease the kubelet actually applies: per-device timeouts when set,
// the kubelet's default otherwise. In particular, when all devices ask for a
// lease longer than the default, the watchdog must not warn before it.
func TestMinHealthCheckTimeout(t *testing.T) {
	for name, tc := range map[string]struct {
		report DeviceHealthReport
		want   time.Duration
	}{
		"empty-report": {
			report: DeviceHealthReport{},
			want:   defaultKubeletHealthTimeout,
		},
		"no-explicit-timeouts": {
			report: DeviceHealthReport{Devices: []DeviceHealth{{}, {}}},
			want:   defaultKubeletHealthTimeout,
		},
		"shorter-than-default": {
			report: DeviceHealthReport{Devices: []DeviceHealth{
				{HealthCheckTimeout: 5 * time.Second},
				{HealthCheckTimeout: time.Minute},
			}},
			want: 5 * time.Second,
		},
		"all-longer-than-default": {
			report: DeviceHealthReport{Devices: []DeviceHealth{
				{HealthCheckTimeout: 2 * time.Minute},
				{HealthCheckTimeout: 3 * time.Minute},
			}},
			want: 2 * time.Minute,
		},
		"unset-device-caps-at-default": {
			report: DeviceHealthReport{Devices: []DeviceHealth{
				{HealthCheckTimeout: 2 * time.Minute},
				{},
			}},
			want: defaultKubeletHealthTimeout,
		},
		"negative-treated-as-default": {
			report: DeviceHealthReport{Devices: []DeviceHealth{
				{HealthCheckTimeout: -time.Second},
			}},
			want: defaultKubeletHealthTimeout,
		},
	} {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, tc.want, minHealthCheckTimeout(tc.report))
		})
	}
}
