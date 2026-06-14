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

package v1beta1

import (
	"context"
	"errors"
	"io"
	"net"
	"path"
	"testing"
	"time"

	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	v1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
)

func sampleResponse() *NodeWatchResourcesResponse {
	return &NodeWatchResourcesResponse{
		Devices: []*DeviceHealth{
			{
				Device:                    &DeviceIdentifier{PoolName: "pool1", DeviceName: "dev1"},
				Health:                    HealthStatus_HEALTHY,
				LastUpdatedTime:           42,
				HealthCheckTimeoutSeconds: 30,
				Message:                   "all good",
			},
			{
				Device:          &DeviceIdentifier{PoolName: "pool2", DeviceName: "dev2"},
				Health:          HealthStatus_UNHEALTHY,
				LastUpdatedTime: 7,
				Message:         "broken",
			},
		},
	}
}

func TestNodeWatchResourcesResponseToV1Alpha1(t *testing.T) {
	in := sampleResponse()
	out := NodeWatchResourcesResponseToV1Alpha1(in)
	if len(out.GetDevices()) != len(in.GetDevices()) {
		t.Fatalf("device count: want %d, got %d", len(in.GetDevices()), len(out.GetDevices()))
	}
	for i, want := range in.GetDevices() {
		got := out.GetDevices()[i]
		if got.GetDevice().GetPoolName() != want.GetDevice().GetPoolName() ||
			got.GetDevice().GetDeviceName() != want.GetDevice().GetDeviceName() {
			t.Errorf("device[%d] identifier mismatch: want %v, got %v", i, want.GetDevice(), got.GetDevice())
		}
		if int32(got.GetHealth()) != int32(want.GetHealth()) {
			t.Errorf("device[%d] health: want %v, got %v", i, want.GetHealth(), got.GetHealth())
		}
		if got.GetLastUpdatedTime() != want.GetLastUpdatedTime() ||
			got.GetHealthCheckTimeoutSeconds() != want.GetHealthCheckTimeoutSeconds() ||
			got.GetMessage() != want.GetMessage() {
			t.Errorf("device[%d] field mismatch: want %+v, got %+v", i, want, got)
		}
	}
}

// TestRoundTrip ensures conversion in both directions is lossless, which is what
// makes serving both versions from a single implementation safe.
func TestRoundTrip(t *testing.T) {
	want := sampleResponse()
	got := NodeWatchResourcesResponseFromV1Alpha1(NodeWatchResourcesResponseToV1Alpha1(want))

	if len(got.GetDevices()) != len(want.GetDevices()) {
		t.Fatalf("device count: want %d, got %d", len(want.GetDevices()), len(got.GetDevices()))
	}
	for i, w := range want.GetDevices() {
		g := got.GetDevices()[i]
		if g.GetDevice().GetPoolName() != w.GetDevice().GetPoolName() ||
			g.GetDevice().GetDeviceName() != w.GetDevice().GetDeviceName() ||
			g.GetHealth() != w.GetHealth() ||
			g.GetLastUpdatedTime() != w.GetLastUpdatedTime() ||
			g.GetHealthCheckTimeoutSeconds() != w.GetHealthCheckTimeoutSeconds() ||
			g.GetMessage() != w.GetMessage() {
			t.Errorf("round-trip changed device[%d]: want %+v, got %+v", i, w, g)
		}
	}
}

func TestHealthStatusValuesMatch(t *testing.T) {
	// The enums must share the same numeric values for the int32 conversion to
	// be correct.
	cases := []struct {
		beta  HealthStatus
		alpha v1alpha1.HealthStatus
	}{
		{HealthStatus_UNKNOWN, v1alpha1.HealthStatus_UNKNOWN},
		{HealthStatus_HEALTHY, v1alpha1.HealthStatus_HEALTHY},
		{HealthStatus_UNHEALTHY, v1alpha1.HealthStatus_UNHEALTHY},
	}
	for _, c := range cases {
		if int32(c.beta) != int32(c.alpha) {
			t.Errorf("enum value mismatch: v1beta1 %v=%d, v1alpha1 %v=%d", c.beta, c.beta, c.alpha, c.alpha)
		}
	}
}

func TestConversionNilSafe(t *testing.T) {
	if got := NodeWatchResourcesResponseToV1Alpha1(nil); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
	if got := NodeWatchResourcesResponseFromV1Alpha1(nil); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
	// A device with no DeviceIdentifier must not panic and must stay nil.
	got := NodeWatchResourcesResponseToV1Alpha1(&NodeWatchResourcesResponse{Devices: []*DeviceHealth{{Health: HealthStatus_UNKNOWN}}})
	if got.GetDevices()[0].GetDevice() != nil {
		t.Errorf("expected nil device identifier, got %v", got.GetDevices()[0].GetDevice())
	}
}

// fakeBetaHealthServer is a minimal v1beta1 health server which sends one
// response and then closes the stream.
type fakeBetaHealthServer struct {
	UnimplementedDRAResourceHealthServer
}

func (f *fakeBetaHealthServer) NodeWatchResources(_ *NodeWatchResourcesRequest, srv DRAResourceHealth_NodeWatchResourcesServer) error {
	return srv.Send(sampleResponse())
}

// TestV1Beta1ServerWrapper verifies that a v1beta1 health server exposed
// through [V1Beta1ServerWrapper] can be consumed by a plain v1alpha1 client
// over a real gRPC connection. This is how the kubeletplugin helper serves
// the older v1alpha1 API to kubelets which do not support v1beta1 yet.
func TestV1Beta1ServerWrapper(t *testing.T) {
	addr := path.Join(t.TempDir(), "drahealth.sock")
	listener, err := net.Listen("unix", addr)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := grpc.NewServer()
	v1alpha1.RegisterDRAResourceHealthServer(server, V1Beta1ServerWrapper{Server: &fakeBetaHealthServer{}})
	go func() {
		_ = server.Serve(listener)
	}()
	defer server.Stop()

	conn, err := grpc.NewClient("unix:"+addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() {
		_ = conn.Close()
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	stream, err := v1alpha1.NewDRAResourceHealthClient(conn).NodeWatchResources(ctx, &v1alpha1.NodeWatchResourcesRequest{})
	if err != nil {
		t.Fatalf("NodeWatchResources: %v", err)
	}

	resp, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	want := NodeWatchResourcesResponseToV1Alpha1(sampleResponse())
	if len(resp.GetDevices()) != len(want.GetDevices()) {
		t.Fatalf("device count: want %d, got %d", len(want.GetDevices()), len(resp.GetDevices()))
	}
	for i, w := range want.GetDevices() {
		g := resp.GetDevices()[i]
		if g.GetDevice().GetPoolName() != w.GetDevice().GetPoolName() ||
			g.GetDevice().GetDeviceName() != w.GetDevice().GetDeviceName() ||
			g.GetHealth() != w.GetHealth() ||
			g.GetLastUpdatedTime() != w.GetLastUpdatedTime() ||
			g.GetHealthCheckTimeoutSeconds() != w.GetHealthCheckTimeoutSeconds() ||
			g.GetMessage() != w.GetMessage() {
			t.Errorf("device[%d] mismatch: want %+v, got %+v", i, w, g)
		}
	}

	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Errorf("expected io.EOF after server closed the stream, got %v", err)
	}
}
