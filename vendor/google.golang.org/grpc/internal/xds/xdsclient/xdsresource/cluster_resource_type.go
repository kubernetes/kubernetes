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

package xdsresource

import (
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/internal/xds/bootstrap"
	xdsclient "google.golang.org/grpc/internal/xds/clients/xdsclient"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource/version"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

const (
	// ClusterResourceTypeName represents the transport agnostic name for the
	// cluster resource.
	ClusterResourceTypeName = "ClusterResource"
)

var (
	// Compile time interface checks.
	_ Type = clusterResourceType{}

	// Singleton instantiation of the resource type implementation.
	clusterType = clusterResourceType{
		resourceTypeState: resourceTypeState{
			typeURL:                    version.V3ClusterURL,
			typeName:                   ClusterResourceTypeName,
			allResourcesRequiredInSotW: true,
		},
	}
)

// clusterResourceType provides the resource-type specific functionality for a
// Cluster resource.
//
// Implements the Type interface.
type clusterResourceType struct {
	resourceTypeState
}

// Decode deserializes and validates an xDS resource serialized inside the
// provided `Any` proto, as received from the xDS management server.
func (clusterResourceType) Decode(opts *DecodeOptions, resource *anypb.Any) (*DecodeResult, error) {
	name, cluster, err := unmarshalClusterResource(resource, opts.ServerConfig)
	switch {
	case name == "":
		// Name is unset only when protobuf deserialization fails.
		return nil, err
	case err != nil:
		// Protobuf deserialization succeeded, but resource validation failed.
		return &DecodeResult{Name: name, Resource: &ClusterResourceData{Resource: ClusterUpdate{}}}, err
	}

	// Perform extra validation here.
	if err := securityConfigValidator(opts.BootstrapConfig, cluster.SecurityCfg); err != nil {
		return &DecodeResult{Name: name, Resource: &ClusterResourceData{Resource: ClusterUpdate{}}}, err
	}

	return &DecodeResult{Name: name, Resource: &ClusterResourceData{Resource: cluster}}, nil

}

// ClusterResourceData wraps the configuration of a Cluster resource as received
// from the management server.
//
// Implements the ResourceData interface.
type ClusterResourceData struct {
	ResourceData

	// TODO: We have always stored update structs by value. See if this can be
	// switched to a pointer?
	Resource ClusterUpdate
}

// RawEqual returns true if other is equal to r.
func (c *ClusterResourceData) RawEqual(other ResourceData) bool {
	if c == nil && other == nil {
		return true
	}
	if (c == nil) != (other == nil) {
		return false
	}
	return proto.Equal(c.Resource.Raw, other.Raw())
}

// ToJSON returns a JSON string representation of the resource data.
func (c *ClusterResourceData) ToJSON() string {
	return pretty.ToJSON(c.Resource)
}

// Raw returns the underlying raw protobuf form of the cluster resource.
func (c *ClusterResourceData) Raw() *anypb.Any {
	return c.Resource.Raw
}

// ClusterWatcher wraps the callbacks to be invoked for different events
// corresponding to the cluster resource being watched. gRFC A88 contains an
// exhaustive list of what method is invoked under what conditions.
type ClusterWatcher interface {
	// ResourceChanged indicates a new version of the resource is available.
	ResourceChanged(resource *ClusterResourceData, done func())

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

type delegatingClusterWatcher struct {
	watcher ClusterWatcher
}

func (d *delegatingClusterWatcher) ResourceChanged(data ResourceData, onDone func()) {
	c := data.(*ClusterResourceData)
	d.watcher.ResourceChanged(c, onDone)
}

func (d *delegatingClusterWatcher) ResourceError(err error, onDone func()) {
	d.watcher.ResourceError(err, onDone)
}

func (d *delegatingClusterWatcher) AmbientError(err error, onDone func()) {
	d.watcher.AmbientError(err, onDone)
}

// WatchCluster uses xDS to discover the configuration associated with the
// provided cluster resource name.
func WatchCluster(p Producer, name string, w ClusterWatcher) (cancel func()) {
	delegator := &delegatingClusterWatcher{watcher: w}
	return p.WatchResource(clusterType, name, delegator)
}

// NewGenericClusterResourceTypeDecoder returns a xdsclient.Decoder that
// wraps the xdsresource.clusterType.
func NewGenericClusterResourceTypeDecoder(bc *bootstrap.Config, gServerCfgMap map[xdsclient.ServerConfig]*bootstrap.ServerConfig) xdsclient.Decoder {
	return &GenericResourceTypeDecoder{ResourceType: clusterType, BootstrapConfig: bc, ServerConfigMap: gServerCfgMap}
}
