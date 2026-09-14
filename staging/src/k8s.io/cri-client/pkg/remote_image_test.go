/*
Copyright 2022 The Kubernetes Authors.

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
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/cri-client/pkg/util"
)

func TestImageServiceSpansWithTP(t *testing.T) {
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
	imgSvc, err := NewRemoteImageServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		WithTracerProvider(tp).
		Build(ctx)
	require.NoError(t, err)
	imgRef, err := imgSvc.PullImage(ctx, &runtimeapi.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, "busybox", imgRef)
	require.NoError(t, err)
	err = tp.ForceFlush(ctx)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.GetSpans())
}

func TestImageServiceSpansWithoutTP(t *testing.T) {
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
	imgSvc, err := NewRemoteImageServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(ctx)
	require.NoError(t, err)
	imgRef, err := imgSvc.PullImage(ctx, &runtimeapi.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, "busybox", imgRef)
	require.NoError(t, err)
	err = tp.ForceFlush(ctx)
	require.NoError(t, err)
	assert.Empty(t, exp.GetSpans())
}

func TestImageServiceBuildValidatesRequiredOptions(t *testing.T) {
	ctx := context.Background()
	_, err := NewRemoteImageServiceBuilder().
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(ctx)
	require.ErrorContains(t, err, "endpoint is required")

	_, err = NewRemoteImageServiceBuilder().
		WithEndpoint("unix:///tmp/cri-client-test.sock").
		Build(ctx)
	require.ErrorContains(t, err, "connectionTimeout must be positive")
}

func TestNewRemoteImageServiceUnixSocketEndpoint(t *testing.T) {
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
	imgSvc, err := NewRemoteImageServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(ctx)
	require.NoError(t, err)
	info, err := imgSvc.ImageFsInfo(ctx)
	require.NoError(t, err)
	assert.NotNil(t, info)
}
