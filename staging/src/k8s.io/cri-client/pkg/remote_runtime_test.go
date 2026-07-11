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

package cri

import (
	"context"
	"errors"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace/noop"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	internalapi "k8s.io/cri-api/pkg/apis"
	kubeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitest "k8s.io/cri-api/pkg/apis/testing"
	fakeremote "k8s.io/cri-client/pkg/fake"
	"k8s.io/cri-client/pkg/util"
)

const defaultConnectionTimeout = 15 * time.Second

// createAndStartFakeRemoteRuntime creates and starts fakeremote.RemoteRuntime.
// It returns the RemoteRuntime, endpoint on success.
// Users should call fakeRuntime.Stop() to cleanup the server.
func createAndStartFakeRemoteRuntime(t *testing.T) (*fakeremote.RemoteRuntime, string) {
	endpoint, err := fakeremote.GenerateEndpoint()
	require.NoError(t, err)

	fakeRuntime := fakeremote.NewFakeRemoteRuntime()
	fakeRuntime.Start(endpoint)

	return fakeRuntime, endpoint
}

func TestGetSpans(t *testing.T) {
	fakeRuntime, endpoint := createAndStartFakeRemoteRuntime(t)
	defer func() {
		fakeRuntime.Stop()
		// clear endpoint file
		if addr, _, err := util.GetAddressAndDialer(endpoint); err == nil {
			if _, err := os.Stat(addr); err == nil {
				os.Remove(addr)
			}
		}
	}()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
	)
	ctx := context.Background()
	rtSvc, err := NewRemoteRuntimeServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		WithTracerProvider(tp).
		Build(ctx)
	require.NoError(t, err)
	_, err = rtSvc.Version(ctx, apitest.FakeVersion)
	require.NoError(t, err)
	err = tp.ForceFlush(ctx)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.GetSpans())
}

func TestVersion(t *testing.T) {
	fakeRuntime, endpoint := createAndStartFakeRemoteRuntime(t)
	defer func() {
		fakeRuntime.Stop()
		// clear endpoint file
		if addr, _, err := util.GetAddressAndDialer(endpoint); err == nil {
			if _, err := os.Stat(addr); err == nil {
				os.Remove(addr)
			}
		}
	}()

	ctx := context.Background()
	rtSvc, err := NewRemoteRuntimeServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(ctx)
	require.NoError(t, err)
	version, err := rtSvc.Version(ctx, apitest.FakeVersion)
	require.NoError(t, err)
	assert.Equal(t, apitest.FakeVersion, version.Version)
	assert.Equal(t, apitest.FakeRuntimeName, version.RuntimeName)
}

func createTestRemoteRuntimeService(t *testing.T) (*fakeremote.RemoteRuntime, internalapi.RuntimeService) {
	t.Helper()
	fakeRuntime, endpoint := createAndStartFakeRemoteRuntime(t)
	rtSvc, err := NewRemoteRuntimeServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(context.Background())
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, rtSvc.Close(context.Background()))
		fakeRuntime.Stop()
		if addr, _, err := util.GetAddressAndDialer(endpoint); err == nil {
			if _, err := os.Stat(addr); err == nil {
				require.NoError(t, os.Remove(addr))
			}
		}
	})
	return fakeRuntime, rtSvc
}

func TestCheckpointPod(t *testing.T) {
	fakeRuntime, rtSvc := createTestRemoteRuntimeService(t)
	request := &kubeapi.CheckpointPodRequest{
		PodSandboxId: "sandbox-id",
		OutputPath:   "/var/lib/kubelet/pod-checkpoints/checkpoint",
		ContainerIds: []string{"container-one", "container-two"},
		Options:      map[string]string{"example.runtime/checkpoint-mode": "fast"},
	}
	deadline := time.Now().Add(time.Minute)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	require.NoError(t, rtSvc.CheckpointPod(ctx, request))
	require.Len(t, fakeRuntime.RuntimeService.CheckpointedPods, 1)
	assert.True(t, proto.Equal(request, fakeRuntime.RuntimeService.CheckpointedPods[0]))
	assert.WithinDuration(t, deadline, fakeRuntime.RuntimeService.CheckpointPodDeadline, time.Second)

	fakeRuntime.RuntimeService.InjectError("CheckpointPod", errors.New("checkpoint failed"))
	require.ErrorContains(t, rtSvc.CheckpointPod(context.Background(), request), "checkpoint failed")
	require.ErrorContains(t, rtSvc.CheckpointPod(context.Background(), nil), "requires non-nil")
}

func TestRestorePod(t *testing.T) {
	fakeRuntime, rtSvc := createTestRemoteRuntimeService(t)
	request := &kubeapi.RestorePodRequest{
		CheckpointPath: "/var/lib/kubelet/pod-checkpoints/checkpoint",
		Config: &kubeapi.PodSandboxConfig{
			Metadata: &kubeapi.PodSandboxMetadata{Name: "pod", Namespace: "default", Uid: "uid"},
		},
		RuntimeHandler: "checkpoint-handler",
		Options:        map[string]string{"example.runtime/restore-mode": "lazy"},
		ContainerConfigs: []*kubeapi.ContainerConfig{
			{Metadata: &kubeapi.ContainerMetadata{Name: "app"}},
		},
	}
	deadline := time.Now().Add(time.Minute)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	response, err := rtSvc.RestorePod(ctx, request)
	require.NoError(t, err)
	assert.Equal(t, "fake-restored-pod-id", response.GetPodSandboxId())
	require.Len(t, response.GetRestoredContainers(), 1)
	assert.Equal(t, "app", response.GetRestoredContainers()[0].GetName())
	assert.NotEmpty(t, response.GetRestoredContainers()[0].GetContainerId())
	require.Len(t, fakeRuntime.RuntimeService.RestoredPods, 1)
	assert.True(t, proto.Equal(request, fakeRuntime.RuntimeService.RestoredPods[0]))
	assert.WithinDuration(t, deadline, fakeRuntime.RuntimeService.RestorePodDeadline, time.Second)

	fakeRuntime.RuntimeService.InjectError("RestorePod", errors.New("restore failed"))
	_, err = rtSvc.RestorePod(context.Background(), request)
	require.ErrorContains(t, err, "restore failed")
	_, err = rtSvc.RestorePod(context.Background(), nil)
	require.ErrorContains(t, err, "requires non-nil")
}

// TestTracerProviderPropagation verifies that the otelgrpc stats handler
// (which provides gRPC client context propagation) is installed when
// WithTracerProvider is not called or is called with a non-nil provider, and
// is skipped when WithTracerProvider(nil) is called explicitly.
func TestTracerProviderPropagation(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exp))
	tracer := tp.Tracer("test")

	cases := []struct {
		name            string
		withTP          func(*RemoteRuntimeServiceBuilder) *RemoteRuntimeServiceBuilder
		wantTraceParent bool
	}{
		{
			name:            "default propagates",
			withTP:          func(b *RemoteRuntimeServiceBuilder) *RemoteRuntimeServiceBuilder { return b },
			wantTraceParent: true,
		},
		{
			name: "noop tracer provider propagates",
			withTP: func(b *RemoteRuntimeServiceBuilder) *RemoteRuntimeServiceBuilder {
				return b.WithTracerProvider(noop.NewTracerProvider())
			},
			wantTraceParent: true,
		},
		{
			name:            "nil tracer provider disables propagation",
			withTP:          func(b *RemoteRuntimeServiceBuilder) *RemoteRuntimeServiceBuilder { return b.WithTracerProvider(nil) },
			wantTraceParent: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			endpoint, sawTraceParent, stop := startCapturingFakeRuntime(t)
			defer stop()

			ctx := context.Background()
			b := NewRemoteRuntimeServiceBuilder().
				WithEndpoint(endpoint).
				WithConnectionTimeout(defaultConnectionTimeout)
			rtSvc, err := tc.withTP(b).Build(ctx)
			require.NoError(t, err)

			rpcCtx, span := tracer.Start(ctx, "outer")
			_, err = rtSvc.Version(rpcCtx, apitest.FakeVersion)
			require.NoError(t, err)
			span.End()

			assert.Equal(t, tc.wantTraceParent, sawTraceParent.Load(), "traceparent header presence")
		})
	}
}

// startCapturingFakeRuntime starts a fake CRI runtime on a fresh endpoint and
// returns a flag that the unary server interceptor flips to true the first
// time it sees a "traceparent" header (the W3C tracecontext header the
// otelgrpc client propagator injects) in incoming gRPC metadata.
func startCapturingFakeRuntime(t *testing.T) (string, *atomic.Bool, func()) {
	endpoint, err := fakeremote.GenerateEndpoint()
	require.NoError(t, err)

	var sawTraceParent atomic.Bool
	interceptor := func(ctx context.Context, req any, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if md, ok := metadata.FromIncomingContext(ctx); ok && len(md.Get("traceparent")) > 0 {
			sawTraceParent.Store(true)
		}
		return handler(ctx, req)
	}

	server := grpc.NewServer(grpc.UnaryInterceptor(interceptor))
	fake := fakeremote.NewFakeRemoteRuntime()
	kubeapi.RegisterRuntimeServiceServer(server, fake)
	kubeapi.RegisterImageServiceServer(server, fake)
	fake.RuntimeService.FakeStatus = &kubeapi.RuntimeStatus{
		Conditions: []*kubeapi.RuntimeCondition{
			{Type: kubeapi.RuntimeReady, Status: true},
			{Type: kubeapi.NetworkReady, Status: true},
		},
	}

	lis, err := util.CreateListener(endpoint)
	require.NoError(t, err)
	go func() { _ = server.Serve(lis) }()

	stop := func() {
		server.Stop()
		if addr, _, err := util.GetAddressAndDialer(endpoint); err == nil {
			if _, err := os.Stat(addr); err == nil {
				if err := os.Remove(addr); err != nil {
					t.Errorf("remove %q: %v", addr, err)
				}
			}
		}
	}
	return endpoint, &sawTraceParent, stop
}

func TestBuildValidatesRequiredOptions(t *testing.T) {
	ctx := context.Background()
	_, err := NewRemoteRuntimeServiceBuilder().
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(ctx)
	require.ErrorContains(t, err, "endpoint is required")

	_, err = NewRemoteRuntimeServiceBuilder().
		WithEndpoint("unix:///tmp/cri-client-test.sock").
		Build(ctx)
	assert.ErrorContains(t, err, "connectionTimeout must be positive")
}

func TestNewRemoteRuntimeServiceUnixSocketEndpoint(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix socket regression test is not applicable on windows")
	}

	fakeRuntime, endpoint := createAndStartFakeRemoteRuntime(t)
	defer func() {
		fakeRuntime.Stop()
		// clear endpoint file
		if addr, _, err := util.GetAddressAndDialer(endpoint); err == nil {
			if _, err := os.Stat(addr); err == nil {
				if err := os.Remove(addr); err != nil {
					t.Errorf("remove %q: %v", addr, err)
				}
			}
		}
	}()

	ctx := context.Background()
	rtSvc, err := NewRemoteRuntimeServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(ctx)
	require.NoError(t, err)
	version, err := rtSvc.Version(ctx, apitest.FakeVersion)
	require.NoError(t, err)
	assert.Equal(t, apitest.FakeVersion, version.Version)
}
