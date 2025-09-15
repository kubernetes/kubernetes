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
	"os"
	"testing"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	internalapi "k8s.io/cri-api/pkg/apis"
	apitest "k8s.io/cri-api/pkg/apis/testing"
	fakeremote "k8s.io/cri-client/pkg/fake"
	"k8s.io/cri-client/pkg/util"
	"k8s.io/klog/v2"
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

func createRemoteRuntimeService(endpoint string, t *testing.T) internalapi.RuntimeService {
	logger := klog.Background()
	runtimeService, err := NewRemoteRuntimeService(endpoint, defaultConnectionTimeout, noop.NewTracerProvider(), &logger)

	require.NoError(t, err)

	return runtimeService
}

func createRemoteRuntimeServiceWithTracerProvider(endpoint string, tp oteltrace.TracerProvider, t *testing.T) internalapi.RuntimeService {
	logger := klog.Background()
	runtimeService, err := NewRemoteRuntimeService(endpoint, defaultConnectionTimeout, tp, &logger)
	require.NoError(t, err)

	return runtimeService
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
	rtSvc := createRemoteRuntimeServiceWithTracerProvider(endpoint, tp, t)
	_, err := rtSvc.Version(ctx, apitest.FakeVersion)
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
	rtSvc := createRemoteRuntimeService(endpoint, t)
	version, err := rtSvc.Version(ctx, apitest.FakeVersion)
	require.NoError(t, err)
	assert.Equal(t, apitest.FakeVersion, version.Version)
	assert.Equal(t, apitest.FakeRuntimeName, version.RuntimeName)
}
