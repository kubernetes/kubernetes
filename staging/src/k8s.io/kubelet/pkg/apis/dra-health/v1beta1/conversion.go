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

// This file provides conversion between the v1beta1 and v1alpha1 DRAResourceHealth
// gRPC APIs together with thin wrappers, modeled after the conversion handling
// for the main DRA gRPC API (k8s.io/kubelet/pkg/apis/dra/v1beta1/conversion.go).
//
// The v1beta1 proto is identical to v1alpha1, so the conversions are mechanical
// field copies. They exist so that a DRA driver only has to implement the newest
// (v1beta1) health server while the kubelet helper can still serve the older
// v1alpha1 API, and so that the kubelet can still consume health from drivers
// which only implement v1alpha1. This enables graceful migration of DRA drivers
// from v1alpha1 to v1beta1: a driver supports both versions and the kubelet
// picks the one it supports.

package v1beta1

import (
	"context"

	grpc "google.golang.org/grpc"

	v1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
)

// --- message conversion ---

func deviceIdentifierToV1Alpha1(in *DeviceIdentifier) *v1alpha1.DeviceIdentifier {
	if in == nil {
		return nil
	}
	return &v1alpha1.DeviceIdentifier{
		PoolName:   in.PoolName,
		DeviceName: in.DeviceName,
	}
}

func deviceIdentifierFromV1Alpha1(in *v1alpha1.DeviceIdentifier) *DeviceIdentifier {
	if in == nil {
		return nil
	}
	return &DeviceIdentifier{
		PoolName:   in.PoolName,
		DeviceName: in.DeviceName,
	}
}

func deviceHealthToV1Alpha1(in *DeviceHealth) *v1alpha1.DeviceHealth {
	if in == nil {
		return nil
	}
	return &v1alpha1.DeviceHealth{
		Device:                    deviceIdentifierToV1Alpha1(in.Device),
		Health:                    v1alpha1.HealthStatus(in.Health),
		LastUpdatedTime:           in.LastUpdatedTime,
		HealthCheckTimeoutSeconds: in.HealthCheckTimeoutSeconds,
		Message:                   in.Message,
	}
}

func deviceHealthFromV1Alpha1(in *v1alpha1.DeviceHealth) *DeviceHealth {
	if in == nil {
		return nil
	}
	return &DeviceHealth{
		Device:                    deviceIdentifierFromV1Alpha1(in.Device),
		Health:                    HealthStatus(in.Health),
		LastUpdatedTime:           in.LastUpdatedTime,
		HealthCheckTimeoutSeconds: in.HealthCheckTimeoutSeconds,
		Message:                   in.Message,
	}
}

// NodeWatchResourcesResponseToV1Alpha1 converts a v1beta1 response to v1alpha1.
func NodeWatchResourcesResponseToV1Alpha1(in *NodeWatchResourcesResponse) *v1alpha1.NodeWatchResourcesResponse {
	if in == nil {
		return nil
	}
	out := &v1alpha1.NodeWatchResourcesResponse{}
	if in.Devices != nil {
		out.Devices = make([]*v1alpha1.DeviceHealth, len(in.Devices))
		for i, d := range in.Devices {
			out.Devices[i] = deviceHealthToV1Alpha1(d)
		}
	}
	return out
}

// NodeWatchResourcesResponseFromV1Alpha1 converts a v1alpha1 response to v1beta1.
func NodeWatchResourcesResponseFromV1Alpha1(in *v1alpha1.NodeWatchResourcesResponse) *NodeWatchResourcesResponse {
	if in == nil {
		return nil
	}
	out := &NodeWatchResourcesResponse{}
	if in.Devices != nil {
		out.Devices = make([]*DeviceHealth, len(in.Devices))
		for i, d := range in.Devices {
			out.Devices[i] = deviceHealthFromV1Alpha1(d)
		}
	}
	return out
}

// --- server wrapper: serve the v1alpha1 API from a v1beta1 implementation ---

// V1Beta1ServerWrapper implements the [v1alpha1.DRAResourceHealthServer]
// interface by wrapping a v1beta1 [DRAResourceHealthServer]. The kubelet helper
// registers it so that a driver which only implements the v1beta1 health server
// also serves the older v1alpha1 API.
type V1Beta1ServerWrapper struct {
	v1alpha1.UnsafeDRAResourceHealthServer
	// Server is the v1beta1 health server provided by the DRA driver.
	Server DRAResourceHealthServer
}

var _ v1alpha1.DRAResourceHealthServer = V1Beta1ServerWrapper{}

func (w V1Beta1ServerWrapper) NodeWatchResources(_ *v1alpha1.NodeWatchResourcesRequest, stream v1alpha1.DRAResourceHealth_NodeWatchResourcesServer) error {
	return w.Server.NodeWatchResources(&NodeWatchResourcesRequest{}, &v1beta1ToV1alpha1ServerStream{ServerStream: stream})
}

// v1beta1ToV1alpha1ServerStream adapts a v1alpha1 server stream so that a
// v1beta1 server implementation can send v1beta1 responses which are converted
// to v1alpha1 before being written to the wire.
type v1beta1ToV1alpha1ServerStream struct {
	grpc.ServerStream
}

var _ DRAResourceHealth_NodeWatchResourcesServer = &v1beta1ToV1alpha1ServerStream{}

func (s *v1beta1ToV1alpha1ServerStream) Send(resp *NodeWatchResourcesResponse) error {
	return s.ServerStream.SendMsg(NodeWatchResourcesResponseToV1Alpha1(resp))
}

// --- client wrapper: consume a v1alpha1 plugin through the v1beta1 client API ---

// V1Alpha1ClientWrapper implements the v1beta1 [DRAResourceHealthClient]
// interface by wrapping a [v1alpha1.DRAResourceHealthClient]. The kubelet uses
// it to consume health updates from a driver which only implements the v1alpha1
// health server, while the rest of the kubelet only deals with v1beta1 types.
type V1Alpha1ClientWrapper struct {
	// Client talks to a plugin that implements the v1alpha1 health server.
	Client v1alpha1.DRAResourceHealthClient
}

var _ DRAResourceHealthClient = V1Alpha1ClientWrapper{}

func (w V1Alpha1ClientWrapper) NodeWatchResources(ctx context.Context, _ *NodeWatchResourcesRequest, opts ...grpc.CallOption) (DRAResourceHealth_NodeWatchResourcesClient, error) {
	stream, err := w.Client.NodeWatchResources(ctx, &v1alpha1.NodeWatchResourcesRequest{}, opts...)
	if err != nil {
		return nil, err
	}
	return &v1alpha1ToV1beta1ClientStream{ClientStream: stream}, nil
}

// v1alpha1ToV1beta1ClientStream adapts a v1alpha1 client stream so that the
// kubelet can receive v1beta1 responses converted from the v1alpha1 messages
// sent by the plugin.
type v1alpha1ToV1beta1ClientStream struct {
	grpc.ClientStream
}

var _ DRAResourceHealth_NodeWatchResourcesClient = &v1alpha1ToV1beta1ClientStream{}

func (s *v1alpha1ToV1beta1ClientStream) Recv() (*NodeWatchResourcesResponse, error) {
	var msg v1alpha1.NodeWatchResourcesResponse
	if err := s.ClientStream.RecvMsg(&msg); err != nil {
		return nil, err
	}
	return NodeWatchResourcesResponseFromV1Alpha1(&msg), nil
}
