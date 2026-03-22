/*
Copyright 2016 The Kubernetes Authors.

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
	"io"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"

	"k8s.io/cri-client/pkg/util"
)

// remoteImageService is a gRPC implementation of internalapi.ImageManagerService.
type remoteImageService struct {
	timeout     time.Duration
	imageClient runtimeapi.ImageServiceClient
	conn        *grpc.ClientConn
	// useStreaming indicates whether to use streaming RPCs for list operations
	// when the CRIListStreaming feature gate is enabled. It falls back to false
	// if a streaming RPC returns Unimplemented. The fallback is racy but kept
	// simple since the worst case is a few extra fallback attempts.
	useStreaming bool
}

// NewRemoteImageService creates a new internalapi.ImageManagerService.
// If useStreaming is true, streaming RPCs will be used for list operations
// instead of unary RPCs. If the runtime returns an Unimplemented error,
// the client automatically falls back to unary RPCs.
// NOTE: useStreaming is supposed to be gated by the CRIListStreaming feature
// gate and is expected to default to true once the feature graduates to GA,
// at which point this parameter may be removed.
func NewRemoteImageService(ctx context.Context, endpoint string, connectionTimeout time.Duration, tp trace.TracerProvider, useStreaming bool) (internalapi.ImageManagerService, error) {
	logger := klog.FromContext(ctx)
	logger.V(3).Info("Connecting to image service", "endpoint", endpoint)
	addr, dialer, err := util.GetAddressAndDialer(endpoint)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, connectionTimeout)
	defer cancel()

	var dialOpts []grpc.DialOption
	dialOpts = append(dialOpts,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithAuthority("localhost"),
		grpc.WithContextDialer(dialer),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize)))
	if tp != nil {
		tracingOpts := []otelgrpc.Option{
			otelgrpc.WithMessageEvents(otelgrpc.ReceivedEvents, otelgrpc.SentEvents),
			otelgrpc.WithPropagators(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{})),
			otelgrpc.WithTracerProvider(tp),
		}
		// Even if there is no TracerProvider, the otelgrpc still handles context propagation.
		// See https://github.com/open-telemetry/opentelemetry-go/tree/main/example/passthrough
		dialOpts = append(dialOpts,
			grpc.WithStatsHandler(otelgrpc.NewClientHandler(tracingOpts...)))
	}

	connParams := grpc.ConnectParams{
		Backoff: backoff.DefaultConfig,
	}
	connParams.MinConnectTimeout = minConnectionTimeout
	connParams.Backoff.BaseDelay = baseBackoffDelay
	connParams.Backoff.MaxDelay = maxBackoffDelay
	dialOpts = append(dialOpts,
		grpc.WithConnectParams(connParams),
	)

	conn, err := grpc.DialContext(ctx, addr, dialOpts...)
	if err != nil {
		logger.Error(err, "Connect remote image service failed", "address", addr)
		return nil, err
	}

	service := &remoteImageService{
		timeout:      connectionTimeout,
		conn:         conn,
		useStreaming: useStreaming,
	}
	if err := service.validateServiceConnection(ctx, conn, endpoint); err != nil {
		return nil, fmt.Errorf("validate service connection: %w", err)
	}

	return service, nil

}

// Close will shutdown the internal gRPC client connection.
func (r *remoteImageService) Close(ctx context.Context) error {
	logger := klog.FromContext(ctx)
	logger.V(3).Info("Closing image service connection")
	return r.conn.Close()
}

// validateServiceConnection tries to connect to the remote image service by
// using the CRI v1 API version and fails if that's not possible.
func (r *remoteImageService) validateServiceConnection(ctx context.Context, conn *grpc.ClientConn, endpoint string) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Validating the CRI v1 API image version")
	r.imageClient = runtimeapi.NewImageServiceClient(conn)

	if _, err := r.imageClient.ImageFsInfo(ctx, &runtimeapi.ImageFsInfoRequest{}); err != nil {
		return fmt.Errorf("validate CRI v1 image API for endpoint %q: %w", endpoint, err)
	}

	logger.V(2).Info("Validated CRI v1 image API")
	return nil
}

// ListImages lists available images.
func (r *remoteImageService) ListImages(ctx context.Context, filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	if r.useStreaming {
		return r.streamImagesV1(ctx, filter)
	}
	return r.listImagesV1(ctx, filter)
}

func (r *remoteImageService) listImagesV1(ctx context.Context, filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	resp, err := r.imageClient.ListImages(ctx, &runtimeapi.ListImagesRequest{
		Filter: filter,
	})
	if err != nil {
		logger := klog.FromContext(ctx)
		logger.Error(err, "ListImages with filter from image service failed", "filter", filter)
		return nil, err
	}

	return resp.Images, nil
}

func (r *remoteImageService) streamImagesV1(ctx context.Context, filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	logger := klog.FromContext(ctx)
	stream, err := r.imageClient.StreamImages(ctx, &runtimeapi.StreamImagesRequest{
		Filter: filter,
	})
	if err != nil {
		logger.Error(err, "StreamImages from image service failed", "filter", filter)
		return nil, err
	}

	var images []*runtimeapi.Image
	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// If the RPC is unimplemented, disable streaming and fall back to the unary RPC.
			// The Unimplemented status is not returned when creating the stream,
			// but when calling Recv() on the stream.
			if status.Code(err) == codes.Unimplemented {
				logger.Info("StreamImages not implemented, falling back to ListImages", "filter", filter)
				r.useStreaming = false
				return r.listImagesV1(ctx, filter)
			}
			logger.Error(err, "StreamImages recv failed", "filter", filter, "itemsReceived", len(images))
			return nil, err
		}
		images = append(images, resp.Images...)
	}

	return images, nil
}

// ImageStatus returns the status of the image.
func (r *remoteImageService) ImageStatus(ctx context.Context, image *runtimeapi.ImageSpec, verbose bool) (*runtimeapi.ImageStatusResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	return r.imageStatusV1(ctx, image, verbose)
}

func (r *remoteImageService) imageStatusV1(ctx context.Context, image *runtimeapi.ImageSpec, verbose bool) (*runtimeapi.ImageStatusResponse, error) {
	resp, err := r.imageClient.ImageStatus(ctx, &runtimeapi.ImageStatusRequest{
		Image:   image,
		Verbose: verbose,
	})
	if err != nil {
		logger := klog.FromContext(ctx)
		logger.Error(err, "Get ImageStatus from image service failed", "image", image.Image)
		return nil, err
	}

	if resp.Image != nil {
		if resp.Image.Id == "" || resp.Image.Size == 0 {
			errorMessage := fmt.Sprintf("Id or size of image %q is not set", image.Image)
			err := errors.New(errorMessage)
			logger := klog.FromContext(ctx)
			logger.Error(err, "ImageStatus failed", "image", image.Image)
			return nil, err
		}
	}

	return resp, nil
}

// PullImage pulls an image with authentication config.
func (r *remoteImageService) PullImage(ctx context.Context, image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	return r.pullImageV1(ctx, image, auth, podSandboxConfig)
}

func (r *remoteImageService) pullImageV1(ctx context.Context, image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	resp, err := r.imageClient.PullImage(ctx, &runtimeapi.PullImageRequest{
		Image:         image,
		Auth:          auth,
		SandboxConfig: podSandboxConfig,
	})
	if err != nil {
		logger := klog.FromContext(ctx)
		logger.Error(err, "PullImage from image service failed", "image", image.Image)

		// We can strip the code from unknown status errors since they add no value
		// and will make them easier to read in the logs/events.
		//
		// It also ensures that checking custom error types from pkg/kubelet/images/types.go
		// works in `imageManager.EnsureImageExists` (pkg/kubelet/images/image_manager.go).
		statusErr, ok := status.FromError(err)
		if ok && statusErr.Code() == codes.Unknown {
			return "", errors.New(statusErr.Message())
		}

		return "", err
	}

	if resp.ImageRef == "" {
		logger := klog.FromContext(ctx)
		logger.Error(errors.New("PullImage failed"), "ImageRef of image is not set", "image", image.Image)
		errorMessage := fmt.Sprintf("imageRef of image %q is not set", image.Image)
		return "", errors.New(errorMessage)
	}

	return resp.ImageRef, nil
}

// RemoveImage removes the image.
func (r *remoteImageService) RemoveImage(ctx context.Context, image *runtimeapi.ImageSpec) (err error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	if _, err = r.imageClient.RemoveImage(ctx, &runtimeapi.RemoveImageRequest{
		Image: image,
	}); err != nil {
		logger := klog.FromContext(ctx)
		logger.Error(err, "RemoveImage from image service failed", "image", image.Image)
		return err
	}

	return nil
}

// ImageFsInfo returns information of the filesystem that is used to store images.
func (r *remoteImageService) ImageFsInfo(ctx context.Context) (*runtimeapi.ImageFsInfoResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	return r.imageFsInfoV1(ctx)
}

func (r *remoteImageService) imageFsInfoV1(ctx context.Context) (*runtimeapi.ImageFsInfoResponse, error) {
	resp, err := r.imageClient.ImageFsInfo(ctx, &runtimeapi.ImageFsInfoRequest{})
	if err != nil {
		logger := klog.FromContext(ctx)
		logger.Error(err, "ImageFsInfo from image service failed")
		return nil, err
	}
	return resp, nil
}
