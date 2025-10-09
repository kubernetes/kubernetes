/*
 *
 * Copyright 2022 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Package xdsresource implements the xDS data model layer.
//
// Provides resource-type specific functionality to unmarshal xDS protos into
// internal data structures that contain only fields gRPC is interested in.
// These internal data structures are passed to components in the xDS stack
// (resolver/balancers/server) that have expressed interest in receiving
// updates to specific resources.
package xdsresource

import (
	"fmt"

	xdsinternal "google.golang.org/grpc/internal/xds"
	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/internal/xds/clients/xdsclient"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource/version"
	"google.golang.org/protobuf/types/known/anypb"
)

func init() {
	xdsinternal.ResourceTypeMapForTesting = make(map[string]any)
	xdsinternal.ResourceTypeMapForTesting[version.V3ListenerURL] = listenerType
	xdsinternal.ResourceTypeMapForTesting[version.V3RouteConfigURL] = routeConfigType
	xdsinternal.ResourceTypeMapForTesting[version.V3ClusterURL] = clusterType
	xdsinternal.ResourceTypeMapForTesting[version.V3EndpointsURL] = endpointsType
}

// Producer contains a single method to discover resource configuration from a
// remote management server using xDS APIs.
//
// The xdsclient package provides a concrete implementation of this interface.
type Producer interface {
	// WatchResource uses xDS to discover the resource associated with the
	// provided resource name. The resource type implementation determines how
	// xDS responses are are deserialized and validated, as received from the
	// xDS management server. Upon receipt of a response from the management
	// server, an appropriate callback on the watcher is invoked.
	WatchResource(rType Type, resourceName string, watcher ResourceWatcher) (cancel func())
}

// ResourceWatcher is notified of the resource updates and errors that are
// received by the xDS client from the management server.
//
// All methods contain a done parameter which should be called when processing
// of the update has completed.  For example, if processing a resource requires
// watching new resources, registration of those new watchers should be
// completed before done is called, which can happen after the ResourceWatcher
// method has returned. Failure to call done will prevent the xDS client from
// providing future ResourceWatcher notifications.
type ResourceWatcher interface {
	// ResourceChanged indicates a new version of the resource is available.
	ResourceChanged(resourceData ResourceData, done func())

	// ResourceError indicates an error occurred while trying to fetch or
	// decode the associated resource. The previous version of the resource
	// should be considered invalid.
	ResourceError(err error, done func())

	// AmbientError indicates an error occurred after a resource has been
	// received that should not modify the use of that resource but may provide
	// useful information about the state of the XDSClient for debugging
	// purposes. The previous version of the resource should still be
	// considered valid.
	AmbientError(err error, done func())
}

// TODO: Once the implementation is complete, rename this interface as
// ResourceType and get rid of the existing ResourceType enum.

// Type wraps all resource-type specific functionality. Each supported resource
// type will provide an implementation of this interface.
type Type interface {
	// TypeURL is the xDS type URL of this resource type for v3 transport.
	TypeURL() string

	// TypeName identifies resources in a transport protocol agnostic way. This
	// can be used for logging/debugging purposes, as well in cases where the
	// resource type name is to be uniquely identified but the actual
	// functionality provided by the resource type is not required.
	//
	// TODO: once Type is renamed to ResourceType, rename TypeName to
	// ResourceTypeName.
	TypeName() string

	// AllResourcesRequiredInSotW indicates whether this resource type requires
	// that all resources be present in every SotW response from the server. If
	// true, a response that does not include a previously seen resource will be
	// interpreted as a deletion of that resource.
	AllResourcesRequiredInSotW() bool

	// Decode deserializes and validates an xDS resource serialized inside the
	// provided `Any` proto, as received from the xDS management server.
	//
	// If protobuf deserialization fails or resource validation fails,
	// returns a non-nil error. Otherwise, returns a fully populated
	// DecodeResult.
	Decode(*DecodeOptions, *anypb.Any) (*DecodeResult, error)
}

// ResourceData contains the configuration data sent by the xDS management
// server, associated with the resource being watched. Every resource type must
// provide an implementation of this interface to represent the configuration
// received from the xDS management server.
type ResourceData interface {
	// RawEqual returns true if the passed in resource data is equal to that of
	// the receiver, based on the underlying raw protobuf message.
	RawEqual(ResourceData) bool

	// ToJSON returns a JSON string representation of the resource data.
	ToJSON() string

	Raw() *anypb.Any
}

// DecodeOptions wraps the options required by ResourceType implementation for
// decoding configuration received from the xDS management server.
type DecodeOptions struct {
	// BootstrapConfig contains the complete bootstrap configuration passed to
	// the xDS client. This contains useful data for resource validation.
	BootstrapConfig *bootstrap.Config
	// ServerConfig contains the server config (from the above bootstrap
	// configuration) of the xDS server from which the current resource, for
	// which Decode() is being invoked, was received.
	ServerConfig *bootstrap.ServerConfig
}

// DecodeResult is the result of a decode operation.
type DecodeResult struct {
	// Name is the name of the resource being watched.
	Name string
	// Resource contains the configuration associated with the resource being
	// watched.
	Resource ResourceData
}

// resourceTypeState wraps the static state associated with concrete resource
// type implementations, which can then embed this struct and get the methods
// implemented here for free.
type resourceTypeState struct {
	typeURL                    string
	typeName                   string
	allResourcesRequiredInSotW bool
}

func (r resourceTypeState) TypeURL() string {
	return r.typeURL
}

func (r resourceTypeState) TypeName() string {
	return r.typeName
}

func (r resourceTypeState) AllResourcesRequiredInSotW() bool {
	return r.allResourcesRequiredInSotW
}

// GenericResourceTypeDecoder wraps an xdsresource.Type and implements
// xdsclient.Decoder.
//
// TODO: #8313 - Delete this once the internal xdsclient usages are updated
// to use the generic xdsclient.ResourceType interface directly.
type GenericResourceTypeDecoder struct {
	ResourceType    Type
	BootstrapConfig *bootstrap.Config
	ServerConfigMap map[xdsclient.ServerConfig]*bootstrap.ServerConfig
}

// Decode deserialize and validate resource bytes of an xDS resource received
// from the xDS management server.
func (gd *GenericResourceTypeDecoder) Decode(resource xdsclient.AnyProto, gOpts xdsclient.DecodeOptions) (*xdsclient.DecodeResult, error) {
	rProto := &anypb.Any{
		TypeUrl: resource.TypeURL,
		Value:   resource.Value,
	}
	opts := &DecodeOptions{BootstrapConfig: gd.BootstrapConfig}
	if gOpts.ServerConfig != nil {
		opts.ServerConfig = gd.ServerConfigMap[*gOpts.ServerConfig]
	}

	result, err := gd.ResourceType.Decode(opts, rProto)
	if result == nil {
		return nil, err
	}
	if err != nil {
		return &xdsclient.DecodeResult{Name: result.Name}, err
	}

	return &xdsclient.DecodeResult{Name: result.Name, Resource: &genericResourceData{resourceData: result.Resource}}, nil
}

// genericResourceData embed an xdsresource.ResourceData and implements
// xdsclient.ResourceData.
//
// TODO: #8313 - Delete this once the internal xdsclient usages are updated
// to use the generic xdsclient.ResourceData interface directly.
type genericResourceData struct {
	resourceData ResourceData
}

// Equal returns true if the passed in xdsclient.ResourceData
// is equal to that of the receiver.
func (grd *genericResourceData) Equal(other xdsclient.ResourceData) bool {
	if other == nil {
		return false
	}
	otherResourceData, ok := other.(*genericResourceData)
	if !ok {
		return false
	}
	return grd.resourceData.RawEqual(otherResourceData.resourceData)
}

// Bytes returns the underlying raw bytes of the wrapped resource.
func (grd *genericResourceData) Bytes() []byte {
	rawAny := grd.resourceData.Raw()
	if rawAny == nil {
		return nil
	}
	return rawAny.Value
}

// genericResourceWatcher wraps xdsresource.ResourceWatcher and implements
// xdsclient.ResourceWatcher.
//
// TODO: #8313 - Delete this once the internal xdsclient usages are updated
// to use the generic xdsclient.ResourceWatcher interface directly.
type genericResourceWatcher struct {
	xdsResourceWatcher ResourceWatcher
}

// ResourceChanged indicates a new version of the wrapped resource is
// available.
func (gw *genericResourceWatcher) ResourceChanged(gData xdsclient.ResourceData, done func()) {
	if gData == nil {
		gw.xdsResourceWatcher.ResourceChanged(nil, done)
		return
	}

	grd, ok := gData.(*genericResourceData)
	if !ok {
		err := fmt.Errorf("genericResourceWatcher received unexpected xdsclient.ResourceData type %T, want *genericResourceData", gData)
		gw.xdsResourceWatcher.ResourceError(err, done)
		return
	}
	gw.xdsResourceWatcher.ResourceChanged(grd.resourceData, done)
}

// ResourceError indicates an error occurred while trying to fetch or
// decode the associated wrapped resource. The previous version of the
// wrapped resource should be considered invalid.
func (gw *genericResourceWatcher) ResourceError(err error, done func()) {
	gw.xdsResourceWatcher.ResourceError(err, done)
}

// AmbientError indicates an error occurred after a resource has been
// received that should not modify the use of that wrapped resource but may
// provide useful information about the state of the XDSClient for debugging
// purposes. The previous version of the wrapped resource should still be
// considered valid.
func (gw *genericResourceWatcher) AmbientError(err error, done func()) {
	gw.xdsResourceWatcher.AmbientError(err, done)
}

// GenericResourceWatcher returns a xdsclient.ResourceWatcher that wraps an
// xdsresource.ResourceWatcher to make it compatible with xdsclient.ResourceWatcher.
func GenericResourceWatcher(xdsResourceWatcher ResourceWatcher) xdsclient.ResourceWatcher {
	if xdsResourceWatcher == nil {
		return nil
	}
	return &genericResourceWatcher{xdsResourceWatcher: xdsResourceWatcher}
}
