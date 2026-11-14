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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	internalapi "k8s.io/cri-api/pkg/apis"
	"k8s.io/cri-client/pkg/metrics"
)

type streamingCounter map[[2]string]int

func (c streamingCounter) Increment(_ context.Context, operation, result string) {
	c[[2]string{operation, result}]++
}

func TestListStreamingMetrics(t *testing.T) {
	totals := streamingCounter{}
	oldTotals := metrics.ListStreamingTotal
	metrics.ListStreamingTotal = totals
	t.Cleanup(func() {
		metrics.ListStreamingTotal = oldTotals
	})

	fakeRuntime, endpoint := createAndStartFakeRemoteRuntime(t)
	t.Cleanup(fakeRuntime.Stop)

	operations := []struct {
		name, label string
		list        func(context.Context, internalapi.RuntimeService, internalapi.ImageManagerService) error
	}{
		{"ListContainers", "list_containers", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListContainers(ctx, nil)
			return err
		}},
		{"ListPodSandbox", "list_podsandbox", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListPodSandbox(ctx, nil)
			return err
		}},
		{"ListContainerStats", "list_container_stats", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListContainerStats(ctx, nil)
			return err
		}},
		{"ListPodSandboxStats", "list_podsandbox_stats", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListPodSandboxStats(ctx, nil)
			return err
		}},
		{"ListPodSandboxMetrics", "list_podsandbox_metrics", func(ctx context.Context, runtime internalapi.RuntimeService, _ internalapi.ImageManagerService) error {
			_, err := runtime.ListPodSandboxMetrics(ctx)
			return err
		}},
		{"ListImages", "list_images", func(ctx context.Context, _ internalapi.RuntimeService, image internalapi.ImageManagerService) error {
			_, err := image.ListImages(ctx, nil)
			return err
		}},
	}
	for _, operation := range operations {
		t.Run(operation.name, func(t *testing.T) {
			for _, tc := range []struct {
				name        string
				streaming   bool
				streamError codes.Code
				unaryError  codes.Code
				wantError   codes.Code
				wantResult  string
			}{
				{name: "disabled"},
				{name: "success", streaming: true, wantResult: "success"},
				{name: "failure", streaming: true, streamError: codes.Internal, wantError: codes.Internal, wantResult: "error"},
				{name: "timeout", streaming: true, streamError: codes.DeadlineExceeded, wantError: codes.DeadlineExceeded, wantResult: "error"},
				{name: "fallback", streaming: true, streamError: codes.Unimplemented, wantResult: "fallback"},
				{name: "failed fallback", streaming: true, streamError: codes.Unimplemented, unaryError: codes.Internal, wantError: codes.Internal, wantResult: "fallback"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					clear(totals)
					ctx := context.Background()
					runtime, err := NewRemoteRuntimeServiceBuilder().WithEndpoint(endpoint).
						WithConnectionTimeout(defaultConnectionTimeout).WithUseStreaming(tc.streaming).Build(ctx)
					require.NoError(t, err)
					t.Cleanup(func() { require.NoError(t, runtime.Close(ctx)) })
					image, err := NewRemoteImageServiceBuilder().WithEndpoint(endpoint).
						WithConnectionTimeout(defaultConnectionTimeout).WithUseStreaming(tc.streaming).Build(ctx)
					require.NoError(t, err)
					t.Cleanup(func() { require.NoError(t, image.Close(ctx)) })

					injectError := fakeRuntime.RuntimeService.InjectError
					if operation.name == "ListImages" {
						injectError = fakeRuntime.ImageService.InjectError
					}
					if tc.streamError != codes.OK {
						injectError(operation.name, status.Error(tc.streamError, "injected streaming error"))
					}
					if tc.unaryError != codes.OK {
						injectError(operation.name, status.Error(tc.unaryError, "injected unary error"))
					}
					assert.Equal(t, tc.wantError, status.Code(operation.list(ctx, runtime, image)))
					want := streamingCounter{}
					if tc.streaming {
						want[[2]string{operation.label, tc.wantResult}] = 1
					}
					assert.Equal(t, want, totals)

					// Cached unary calls do not add attempts; streaming retries do.
					require.NoError(t, operation.list(ctx, runtime, image))
					if tc.streaming && tc.wantResult != "fallback" {
						want[[2]string{operation.label, "success"}]++
					}
					assert.Equal(t, want, totals)
				})
			}
		})
	}
}
