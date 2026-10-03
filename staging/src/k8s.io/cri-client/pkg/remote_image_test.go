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
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
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

func TestPullSecurityProfile(t *testing.T) {
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

	const ref = "registry.example.com/profile@sha256:0000000000000000000000000000000000000000000000000000000000000000"
	image := &runtimeapi.ImageSpec{Image: ref}
	auth := &runtimeapi.AuthConfig{Username: "user", Password: "pass"}

	resp, err := imgSvc.PullSecurityProfile(ctx, image, auth, nil, runtimeapi.SecurityProfileKind_Seccomp)
	require.NoError(t, err)
	assert.False(t, resp.Cached)
	fakeRuntime.ImageService.AssertSecurityProfilePulledWithAuth(t, image, auth, runtimeapi.SecurityProfileKind_Seccomp, "first pull")
	fakeRuntime.ImageService.Lock()
	assert.Equal(t, runtimeapi.SecurityProfileKind_Seccomp, fakeRuntime.ImageService.SecurityProfiles[ref])
	fakeRuntime.ImageService.Unlock()

	resp, err = imgSvc.PullSecurityProfile(ctx, image, nil, nil, runtimeapi.SecurityProfileKind_Seccomp)
	require.NoError(t, err)
	assert.True(t, resp.Cached)

	for _, tc := range []struct {
		name         string
		injected     error
		expectedErr  string
		expectedCode codes.Code
	}{{
		name:        "unknown code is stripped",
		injected:    errors.New("boom"),
		expectedErr: "boom",
	}, {
		name:        "unknown code with a well-known prefix is stripped",
		injected:    fmt.Errorf("%w: wrong config media type", crierrors.ErrSecurityProfileInvalid),
		expectedErr: "SecurityProfileInvalid: wrong config media type",
	}, {
		name:        "SecurityProfileInvalid with another code is stripped",
		injected:    status.Errorf(codes.InvalidArgument, "%s: layer count 2", crierrors.ErrSecurityProfileInvalid),
		expectedErr: "SecurityProfileInvalid: layer count 2",
	}, {
		name:        "RegistryUnavailable with another code is stripped",
		injected:    status.Errorf(codes.Unavailable, "%s: connection refused", crierrors.ErrRegistryUnavailable),
		expectedErr: "RegistryUnavailable: connection refused",
	}, {
		name:        "SignatureValidationFailed with another code is stripped",
		injected:    status.Errorf(codes.PermissionDenied, "%s: no signature", crierrors.ErrSignatureValidationFailed),
		expectedErr: "SignatureValidationFailed: no signature",
	}, {
		name:         "other codes are kept",
		injected:     status.Error(codes.Unimplemented, "unknown method PullSecurityProfile"),
		expectedCode: codes.Unimplemented,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			fakeRuntime.ImageService.InjectError("PullSecurityProfile", tc.injected)
			_, err := imgSvc.PullSecurityProfile(ctx, image, nil, nil, runtimeapi.SecurityProfileKind_Seccomp)
			if tc.expectedErr != "" {
				require.EqualError(t, err, tc.expectedErr)
			} else {
				require.Equal(t, tc.expectedCode, status.Code(err))
			}
		})
	}

	// Unsupported kinds are rejected permanently.
	for _, kind := range []runtimeapi.SecurityProfileKind{runtimeapi.SecurityProfileKind_SecurityProfileKindUnspecified, runtimeapi.SecurityProfileKind_AppArmor} {
		_, err = imgSvc.PullSecurityProfile(ctx, image, nil, nil, kind)
		require.Error(t, err, kind.String())
		require.True(t, strings.HasPrefix(err.Error(), crierrors.ErrSecurityProfileInvalid.Error()), kind.String())
	}
}
