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

package v1

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"path"
	"testing"
	"time"

	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

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

func TestHealthStatusValuesMatch(t *testing.T) {
	// The enums must share the same numeric values for the int32 conversion to
	// be correct.
	cases := []struct {
		v1    HealthStatus
		alpha v1alpha1.HealthStatus
	}{
		{HealthStatus_UNKNOWN, v1alpha1.HealthStatus_UNKNOWN},
		{HealthStatus_HEALTHY, v1alpha1.HealthStatus_HEALTHY},
		{HealthStatus_UNHEALTHY, v1alpha1.HealthStatus_UNHEALTHY},
	}
	for _, c := range cases {
		if int32(c.v1) != int32(c.alpha) {
			t.Errorf("enum value mismatch: v1 %v=%d, v1alpha1 %v=%d", c.v1, c.v1, c.alpha, c.alpha)
		}
	}
}

// requireSameSchema fails when the v1 and v1alpha1 message descriptors have
// diverged: a field added to one proto but not the other, or with a different
// number, name, kind, or cardinality. The v1alpha1 serving support relies on
// the two protos staying wire-identical.
func requireSameSchema(t *testing.T, a, b protoreflect.MessageDescriptor) {
	t.Helper()
	aFields, bFields := a.Fields(), b.Fields()
	for i := 0; i < aFields.Len(); i++ {
		fd := aFields.Get(i)
		other := bFields.ByNumber(fd.Number())
		if other == nil {
			t.Fatalf("field %d (%s) of %s has no counterpart in %s", fd.Number(), fd.Name(), a.FullName(), b.FullName())
		}
		if fd.Name() != other.Name() || fd.Kind() != other.Kind() || fd.Cardinality() != other.Cardinality() {
			t.Fatalf("field %d differs between %s (%s %s %s) and %s (%s %s %s)",
				fd.Number(),
				a.FullName(), fd.Name(), fd.Kind(), fd.Cardinality(),
				b.FullName(), other.Name(), other.Kind(), other.Cardinality())
		}
		if fd.Kind() == protoreflect.MessageKind {
			requireSameSchema(t, fd.Message(), other.Message())
		}
	}
	if aFields.Len() != bFields.Len() {
		t.Fatalf("%s has %d fields, %s has %d", a.FullName(), aFields.Len(), b.FullName(), bFields.Len())
	}
}

// TestSchemasMatch catches a field being added to the v1 proto without also
// being added to v1alpha1 (or vice versa).
func TestSchemasMatch(t *testing.T) {
	requireSameSchema(t,
		(&NodeWatchResourcesRequest{}).ProtoReflect().Descriptor(),
		(&v1alpha1.NodeWatchResourcesRequest{}).ProtoReflect().Descriptor())
	requireSameSchema(t,
		(&NodeWatchResourcesResponse{}).ProtoReflect().Descriptor(),
		(&v1alpha1.NodeWatchResourcesResponse{}).ProtoReflect().Descriptor())
}

// fillMessage sets every field of the message to a non-zero value via
// reflection, recursing into nested and repeated messages. Because it
// discovers fields dynamically, a field added to the proto later is filled
// automatically without updating this test.
func fillMessage(t *testing.T, m protoreflect.Message) {
	t.Helper()
	fields := m.Descriptor().Fields()
	for i := 0; i < fields.Len(); i++ {
		fd := fields.Get(i)
		switch {
		case fd.IsMap():
			t.Fatalf("map field %s not supported by fillMessage, extend it", fd.FullName())
		case fd.IsList():
			list := m.Mutable(fd).List()
			v := list.NewElement()
			if fd.Kind() == protoreflect.MessageKind {
				fillMessage(t, v.Message())
			} else {
				v = nonZeroScalar(t, fd)
			}
			list.Append(v)
		case fd.Kind() == protoreflect.MessageKind:
			fillMessage(t, m.Mutable(fd).Message())
		default:
			m.Set(fd, nonZeroScalar(t, fd))
		}
	}
}

func nonZeroScalar(t *testing.T, fd protoreflect.FieldDescriptor) protoreflect.Value {
	t.Helper()
	switch fd.Kind() {
	case protoreflect.BoolKind:
		return protoreflect.ValueOfBool(true)
	case protoreflect.StringKind:
		return protoreflect.ValueOfString(fmt.Sprintf("value-%d", fd.Number()))
	case protoreflect.BytesKind:
		return protoreflect.ValueOfBytes([]byte{byte(fd.Number())})
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		return protoreflect.ValueOfInt32(int32(fd.Number()) + 1)
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		return protoreflect.ValueOfInt64(int64(fd.Number()) + 1)
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		return protoreflect.ValueOfUint32(uint32(fd.Number()) + 1)
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		return protoreflect.ValueOfUint64(uint64(fd.Number()) + 1)
	case protoreflect.FloatKind:
		return protoreflect.ValueOfFloat32(float32(fd.Number()) + 0.5)
	case protoreflect.DoubleKind:
		return protoreflect.ValueOfFloat64(float64(fd.Number()) + 0.5)
	case protoreflect.EnumKind:
		values := fd.Enum().Values()
		return protoreflect.ValueOfEnum(values.Get(values.Len() - 1).Number())
	default:
		t.Fatalf("field %s has kind %s not supported by fillMessage, extend it", fd.FullName(), fd.Kind())
		return protoreflect.Value{}
	}
}

// TestConversionFidelity catches the conversion code dropping or garbling a
// field. Every field of the v1 response gets a non-zero value via reflection,
// so a field added to both protos but forgotten in the hand-written conversion
// makes this test fail: the v1 and v1alpha1 protos are wire-identical, so a
// lossless conversion must marshal to exactly the same bytes as the input.
// Both directions are checked.
func TestConversionFidelity(t *testing.T) {
	in := &NodeWatchResourcesResponse{}
	fillMessage(t, in.ProtoReflect())

	out := NodeWatchResourcesResponseToV1Alpha1(in)
	back := NodeWatchResourcesResponseFromV1Alpha1(out)

	opts := proto.MarshalOptions{Deterministic: true}
	inBytes, err := opts.Marshal(in)
	if err != nil {
		t.Fatalf("marshal v1: %v", err)
	}
	outBytes, err := opts.Marshal(out)
	if err != nil {
		t.Fatalf("marshal v1alpha1: %v", err)
	}
	backBytes, err := opts.Marshal(back)
	if err != nil {
		t.Fatalf("marshal round-tripped v1: %v", err)
	}
	if string(inBytes) != string(outBytes) {
		t.Errorf("conversion to v1alpha1 is not lossless:\nv1 input: %v\nv1alpha1 output: %v", in, out)
	}
	if string(inBytes) != string(backBytes) {
		t.Errorf("conversion from v1alpha1 is not lossless:\nv1 input: %v\nround-tripped v1: %v", in, back)
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

// fakeV1HealthServer is a minimal v1 health server which sends one
// response and then closes the stream.
type fakeV1HealthServer struct {
	UnimplementedDRAResourceHealthServer
}

func (f *fakeV1HealthServer) NodeWatchResources(_ *NodeWatchResourcesRequest, srv DRAResourceHealth_NodeWatchResourcesServer) error {
	return srv.Send(sampleResponse())
}

// TestV1ServerWrapper verifies that a v1 health server exposed
// through [V1ServerWrapper] can be consumed by a plain v1alpha1 client
// over a real gRPC connection. This is how the kubeletplugin helper serves
// the older v1alpha1 API to kubelets which do not support v1 yet.
func TestV1ServerWrapper(t *testing.T) {
	addr := path.Join(t.TempDir(), "drahealth.sock")
	listener, err := net.Listen("unix", addr)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := grpc.NewServer()
	v1alpha1.RegisterDRAResourceHealthServer(server, V1ServerWrapper{Server: &fakeV1HealthServer{}})
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
