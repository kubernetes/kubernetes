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
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
	fakeremote "k8s.io/cri-client/pkg/fake"
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

// newRemoteImageService starts a fake remote runtime and returns it together
// with an image service connected to it. Both are stopped when the test ends.
func newRemoteImageService(t *testing.T) (*fakeremote.RemoteRuntime, internalapi.ImageManagerService) {
	t.Helper()
	fakeRuntime, endpoint := createAndStartFakeRemoteRuntime(t)
	t.Cleanup(func() {
		fakeRuntime.Stop()
		// clear endpoint file
		if addr, _, err := util.GetAddressAndDialer(endpoint); err == nil {
			if _, err := os.Stat(addr); err == nil {
				if err := os.Remove(addr); err != nil {
					t.Errorf("remove %q: %v", addr, err)
				}
			}
		}
	})

	imgSvc, err := NewRemoteImageServiceBuilder().
		WithEndpoint(endpoint).
		WithConnectionTimeout(defaultConnectionTimeout).
		Build(context.Background())
	require.NoError(t, err)
	t.Cleanup(func() {
		if err := imgSvc.Close(context.Background()); err != nil {
			t.Errorf("close image service: %v", err)
		}
	})

	return fakeRuntime, imgSvc
}

func TestPullSecurityProfile(t *testing.T) {
	fakeRuntime, imgSvc := newRemoteImageService(t)
	ctx := context.Background()

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

	// Unknown codes are stripped like for PullImage, other codes are kept
	// with their message, which callers match by its well-known prefix.
	for _, tc := range []struct {
		name            string
		injected        error
		expectedErr     string
		expectedCode    codes.Code
		expectedMessage string
	}{{
		name:        "unknown code is stripped",
		injected:    errors.New("boom"),
		expectedErr: "boom",
	}, {
		name:        "unknown code with a well-known prefix is stripped",
		injected:    fmt.Errorf("%w: wrong config media type", crierrors.ErrSecurityProfileInvalid),
		expectedErr: "SecurityProfileInvalid: wrong config media type",
	}, {
		name:            "SecurityProfileInvalid with another code keeps it",
		injected:        status.Errorf(codes.InvalidArgument, "%s: layer count 2", crierrors.ErrSecurityProfileInvalid),
		expectedCode:    codes.InvalidArgument,
		expectedMessage: "SecurityProfileInvalid: layer count 2",
	}, {
		name:            "RegistryUnavailable with another code keeps it",
		injected:        status.Errorf(codes.Unavailable, "%s: connection refused", crierrors.ErrRegistryUnavailable),
		expectedCode:    codes.Unavailable,
		expectedMessage: "RegistryUnavailable: connection refused",
	}, {
		name:            "other codes are kept",
		injected:        status.Error(codes.Unimplemented, "unknown method PullSecurityProfile"),
		expectedCode:    codes.Unimplemented,
		expectedMessage: "unknown method PullSecurityProfile",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			fakeRuntime.ImageService.InjectError("PullSecurityProfile", tc.injected)
			_, err := imgSvc.PullSecurityProfile(ctx, image, nil, nil, runtimeapi.SecurityProfileKind_Seccomp)
			if tc.expectedErr != "" {
				require.EqualError(t, err, tc.expectedErr)
			} else {
				require.Equal(t, tc.expectedCode, status.Code(err))
				require.Equal(t, tc.expectedMessage, status.Convert(err).Message())
			}
		})
	}

	// Unsupported kinds and references without a digest are rejected
	// permanently, with the code the proto asks for.
	for _, tc := range []struct {
		name  string
		image *runtimeapi.ImageSpec
		kind  runtimeapi.SecurityProfileKind
	}{
		{name: "unspecified kind", image: image, kind: runtimeapi.SecurityProfileKind_SecurityProfileKindUnspecified},
		{name: "unknown kind", image: image, kind: runtimeapi.SecurityProfileKind(42)},
		{name: "tag", image: &runtimeapi.ImageSpec{Image: "registry.example.com/profile:latest"}, kind: runtimeapi.SecurityProfileKind_Seccomp},
		{name: "no image", image: nil, kind: runtimeapi.SecurityProfileKind_Seccomp},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := imgSvc.PullSecurityProfile(ctx, tc.image, nil, nil, tc.kind)
			require.Equal(t, codes.InvalidArgument, status.Code(err))
			require.True(t, strings.HasPrefix(status.Convert(err).Message(), crierrors.ErrSecurityProfileInvalid.Error()), err.Error())
		})
	}
}

func TestListAndRemoveSecurityProfiles(t *testing.T) {
	fakeRuntime, imgSvc := newRemoteImageService(t)
	ctx := context.Background()

	const (
		digestA = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
		digestB = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
		refA    = "registry.example.com/a@" + digestA
		mirrorA = "mirror.example.com/a@" + digestA
		refB    = "registry.example.com/b@" + digestB
	)
	fakeRuntime.ImageService.SetFakeImageSize(42)
	for _, ref := range []string{refA, mirrorA, refB} {
		_, err := imgSvc.PullSecurityProfile(ctx, &runtimeapi.ImageSpec{Image: ref}, nil, nil, runtimeapi.SecurityProfileKind_Seccomp)
		require.NoError(t, err)
	}

	profiles, err := imgSvc.ListSecurityProfiles(ctx)
	require.NoError(t, err)
	require.Len(t, profiles, 2)
	assert.Equal(t, digestA, profiles[0].Digest)
	assert.Equal(t, []string{mirrorA, refA}, profiles[0].Refs)
	assert.Equal(t, uint64(42), profiles[0].Size)
	assert.Equal(t, digestB, profiles[1].Digest)

	// Removing a digest removes all of its references, and removing it again
	// is not an error. An empty digest is rejected.
	require.Equal(t, codes.InvalidArgument, status.Code(imgSvc.RemoveSecurityProfile(ctx, "")))
	require.NoError(t, imgSvc.RemoveSecurityProfile(ctx, digestA))
	require.NoError(t, imgSvc.RemoveSecurityProfile(ctx, digestA))
	profiles, err = imgSvc.ListSecurityProfiles(ctx)
	require.NoError(t, err)
	require.Len(t, profiles, 1)
	assert.Equal(t, digestB, profiles[0].Digest)

	fakeRuntime.ImageService.InjectError("ListSecurityProfiles", status.Error(codes.Unimplemented, "unknown method ListSecurityProfiles"))
	_, err = imgSvc.ListSecurityProfiles(ctx)
	require.Equal(t, codes.Unimplemented, status.Code(err))
}
