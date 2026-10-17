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

package cri

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	remote "k8s.io/cri-client/pkg"
	fakeremote "k8s.io/cri-client/pkg/fake"
	criutil "k8s.io/cri-client/pkg/util"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

func TestCRIStreamingMetric(t *testing.T) {
	operations := []struct {
		name    string
		service string
		list    func(context.Context, internalapi.RuntimeService, internalapi.ImageManagerService) error
	}{
		{"ListContainers", "runtime", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListContainers(ctx, nil)
			return err
		}},
		{"ListPodSandbox", "runtime", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListPodSandbox(ctx, nil)
			return err
		}},
		{"ListContainerStats", "runtime", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListContainerStats(ctx, nil)
			return err
		}},
		{"ListPodSandboxStats", "runtime", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListPodSandboxStats(ctx, nil)
			return err
		}},
		{"ListPodSandboxMetrics", "runtime", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListPodSandboxMetrics(ctx)
			return err
		}},
		{"ListImages", "image", func(ctx context.Context, _ internalapi.RuntimeService, image internalapi.ImageManagerService) error {
			_, err := image.ListImages(ctx, nil)
			return err
		}},
	}

	for _, enabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("feature gate enabled=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CRIListStreaming, enabled)
			for _, operation := range operations {
				t.Run(operation.name, func(t *testing.T) {
					ctx := ktesting.Init(t)
					registry := compbasemetrics.NewKubeRegistry()
					// Register even with the gate disabled to detect accidental updates.
					registry.MustRegister(metrics.CRIStreamingEnabled)
					metrics.CRIStreamingEnabled.Reset()
					t.Cleanup(metrics.CRIStreamingEnabled.Reset)

					var streamError atomic.Int32
					var streamCalls, unaryCalls atomic.Int32
					server := grpc.NewServer(
						grpc.StreamInterceptor(func(srv interface{}, stream grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
							streamCalls.Add(1)
							if code := codes.Code(streamError.Load()); code != codes.OK {
								return status.Error(code, "injected streaming error")
							}
							return handler(srv, stream)
						}),
						grpc.UnaryInterceptor(func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
							if strings.HasSuffix(info.FullMethod, "/"+operation.name) {
								unaryCalls.Add(1)
							}
							return handler(ctx, req)
						}),
					)
					t.Cleanup(server.Stop)
					fake := fakeremote.NewFakeRemoteRuntime()
					t.Cleanup(fake.Stop)
					runtimeapi.RegisterRuntimeServiceServer(server, fake)
					runtimeapi.RegisterImageServiceServer(server, fake)
					endpoint, err := fakeremote.GenerateEndpoint()
					require.NoError(t, err)
					listener, err := criutil.CreateListener(endpoint)
					require.NoError(t, err)
					go server.Serve(listener)

					runtime, err := remote.NewRemoteRuntimeServiceBuilder().
						WithEndpoint(endpoint).
						WithConnectionTimeout(5 * time.Second).
						WithUseStreaming(enabled).
						Build(ctx)
					require.NoError(t, err)
					t.Cleanup(func() { require.NoError(t, runtime.Close(ctx)) })
					image, err := remote.NewRemoteImageServiceBuilder().
						WithEndpoint(endpoint).
						WithConnectionTimeout(5 * time.Second).
						WithUseStreaming(enabled).
						Build(ctx)
					require.NoError(t, err)
					t.Cleanup(func() { require.NoError(t, image.Close(ctx)) })
					runtime = NewInstrumentedRuntimeService(runtime)
					image = NewInstrumentedImageManagerService(image)

					const metricName = "kubelet_cri_streaming_enabled"
					checkGauge := func(value int) {
						t.Helper()
						var expected string
						if enabled {
							expected = fmt.Sprintf(`# HELP kubelet_cri_streaming_enabled [ALPHA] Whether CRI list streaming RPCs are active (1) or fell back to unary (0).
# TYPE kubelet_cri_streaming_enabled gauge
kubelet_cri_streaming_enabled{service=%q} %d
`, operation.service, value)
						}
						require.NoError(t, testutil.GatherAndCompare(registry, strings.NewReader(expected), metricName))
					}

					require.NoError(t, testutil.GatherAndCompare(registry, strings.NewReader(""), metricName))
					require.NoError(t, operation.list(ctx, runtime, image))
					checkGauge(1)
					if !enabled {
						require.Equal(t, int32(0), streamCalls.Load())
						require.Equal(t, int32(1), unaryCalls.Load())
						return
					}
					require.Equal(t, int32(1), streamCalls.Load())
					require.Equal(t, int32(0), unaryCalls.Load())

					// The gauge reports selection, including when the selected RPC fails.
					streamError.Store(int32(codes.Internal))
					err = operation.list(ctx, runtime, image)
					require.Equal(t, codes.Internal, status.Code(err))
					checkGauge(1)

					streamError.Store(int32(codes.Unimplemented))
					require.NoError(t, operation.list(ctx, runtime, image))
					checkGauge(0)
					require.Equal(t, int32(3), streamCalls.Load())
					require.Equal(t, int32(1), unaryCalls.Load())

					// Restoring runtime support must not clear the client's cached fallback.
					streamError.Store(int32(codes.OK))
					require.NoError(t, operation.list(ctx, runtime, image))
					checkGauge(0)
					require.Equal(t, int32(3), streamCalls.Load())
					require.Equal(t, int32(2), unaryCalls.Load())
				})
			}
		})
	}
}
