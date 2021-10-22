// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// AUTO-GENERATED CODE. DO NOT EDIT.

package container

import (
	"context"
	"time"

	"cloud.google.com/go/internal/version"
	gax "github.com/googleapis/gax-go/v2"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
	containerpb "google.golang.org/genproto/googleapis/container/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

// ClusterManagerCallOptions contains the retry settings for each method of ClusterManagerClient.
type ClusterManagerCallOptions struct {
	ListClusters            []gax.CallOption
	GetCluster              []gax.CallOption
	CreateCluster           []gax.CallOption
	UpdateCluster           []gax.CallOption
	UpdateNodePool          []gax.CallOption
	SetNodePoolAutoscaling  []gax.CallOption
	SetLoggingService       []gax.CallOption
	SetMonitoringService    []gax.CallOption
	SetAddonsConfig         []gax.CallOption
	SetLocations            []gax.CallOption
	UpdateMaster            []gax.CallOption
	SetMasterAuth           []gax.CallOption
	DeleteCluster           []gax.CallOption
	ListOperations          []gax.CallOption
	GetOperation            []gax.CallOption
	CancelOperation         []gax.CallOption
	GetServerConfig         []gax.CallOption
	ListNodePools           []gax.CallOption
	GetNodePool             []gax.CallOption
	CreateNodePool          []gax.CallOption
	DeleteNodePool          []gax.CallOption
	RollbackNodePoolUpgrade []gax.CallOption
	SetNodePoolManagement   []gax.CallOption
	SetLabels               []gax.CallOption
	SetLegacyAbac           []gax.CallOption
	StartIPRotation         []gax.CallOption
	CompleteIPRotation      []gax.CallOption
	SetNodePoolSize         []gax.CallOption
	SetNetworkPolicy        []gax.CallOption
	SetMaintenancePolicy    []gax.CallOption
}

func defaultClusterManagerClientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint("container.googleapis.com:443"),
		option.WithScopes(DefaultAuthScopes()...),
	}
}

func defaultClusterManagerCallOptions() *ClusterManagerCallOptions {
	retry := map[[2]string][]gax.CallOption{
		{"default", "idempotent"}: {
			gax.WithRetry(func() gax.Retryer {
				return gax.OnCodes([]codes.Code{
					codes.DeadlineExceeded,
					codes.Unavailable,
				}, gax.Backoff{
					Initial:    100 * time.Millisecond,
					Max:        60000 * time.Millisecond,
					Multiplier: 1.3,
				})
			}),
		},
	}
	return &ClusterManagerCallOptions{
		ListClusters:            retry[[2]string{"default", "idempotent"}],
		GetCluster:              retry[[2]string{"default", "idempotent"}],
		CreateCluster:           retry[[2]string{"default", "non_idempotent"}],
		UpdateCluster:           retry[[2]string{"default", "non_idempotent"}],
		UpdateNodePool:          retry[[2]string{"default", "non_idempotent"}],
		SetNodePoolAutoscaling:  retry[[2]string{"default", "non_idempotent"}],
		SetLoggingService:       retry[[2]string{"default", "non_idempotent"}],
		SetMonitoringService:    retry[[2]string{"default", "non_idempotent"}],
		SetAddonsConfig:         retry[[2]string{"default", "non_idempotent"}],
		SetLocations:            retry[[2]string{"default", "non_idempotent"}],
		UpdateMaster:            retry[[2]string{"default", "non_idempotent"}],
		SetMasterAuth:           retry[[2]string{"default", "non_idempotent"}],
		DeleteCluster:           retry[[2]string{"default", "idempotent"}],
		ListOperations:          retry[[2]string{"default", "idempotent"}],
		GetOperation:            retry[[2]string{"default", "idempotent"}],
		CancelOperation:         retry[[2]string{"default", "non_idempotent"}],
		GetServerConfig:         retry[[2]string{"default", "idempotent"}],
		ListNodePools:           retry[[2]string{"default", "idempotent"}],
		GetNodePool:             retry[[2]string{"default", "idempotent"}],
		CreateNodePool:          retry[[2]string{"default", "non_idempotent"}],
		DeleteNodePool:          retry[[2]string{"default", "idempotent"}],
		RollbackNodePoolUpgrade: retry[[2]string{"default", "non_idempotent"}],
		SetNodePoolManagement:   retry[[2]string{"default", "non_idempotent"}],
		SetLabels:               retry[[2]string{"default", "non_idempotent"}],
		SetLegacyAbac:           retry[[2]string{"default", "non_idempotent"}],
		StartIPRotation:         retry[[2]string{"default", "non_idempotent"}],
		CompleteIPRotation:      retry[[2]string{"default", "non_idempotent"}],
		SetNodePoolSize:         retry[[2]string{"default", "non_idempotent"}],
		SetNetworkPolicy:        retry[[2]string{"default", "non_idempotent"}],
		SetMaintenancePolicy:    retry[[2]string{"default", "non_idempotent"}],
	}
}

// ClusterManagerClient is a client for interacting with Google Container Engine API.
//
// Methods, except Close, may be called concurrently. However, fields must not be modified concurrently with method calls.
type ClusterManagerClient struct {
	// The connection to the service.
	conn *grpc.ClientConn

	// The gRPC API client.
	clusterManagerClient containerpb.ClusterManagerClient

	// The call options for this service.
	CallOptions *ClusterManagerCallOptions

	// The x-goog-* metadata to be sent with each request.
	xGoogMetadata metadata.MD
}

// NewClusterManagerClient creates a new cluster manager client.
//
// Google Container Engine Cluster Manager v1
func NewClusterManagerClient(ctx context.Context, opts ...option.ClientOption) (*ClusterManagerClient, error) {
	conn, err := transport.DialGRPC(ctx, append(defaultClusterManagerClientOptions(), opts...)...)
	if err != nil {
		return nil, err
	}
	c := &ClusterManagerClient{
		conn:        conn,
		CallOptions: defaultClusterManagerCallOptions(),

		clusterManagerClient: containerpb.NewClusterManagerClient(conn),
	}
	c.setGoogleClientInfo()
	return c, nil
}

// Connection returns the client's connection to the API service.
func (c *ClusterManagerClient) Connection() *grpc.ClientConn {
	return c.conn
}

// Close closes the connection to the API service. The user should invoke this when
// the client is no longer required.
func (c *ClusterManagerClient) Close() error {
	return c.conn.Close()
}

// setGoogleClientInfo sets the name and version of the application in
// the `x-goog-api-client` header passed on each request. Intended for
// use by Google-written clients.
func (c *ClusterManagerClient) setGoogleClientInfo(keyval ...string) {
	kv := append([]string{"gl-go", version.Go()}, keyval...)
	kv = append(kv, "gapic", version.Repo, "gax", gax.Version, "grpc", grpc.Version)
	c.xGoogMetadata = metadata.Pairs("x-goog-api-client", gax.XGoogHeader(kv...))
}

// ListClusters lists all clusters owned by a project in either the specified zone or all
// zones.
func (c *ClusterManagerClient) ListClusters(ctx context.Context, req *containerpb.ListClustersRequest, opts ...gax.CallOption) (*containerpb.ListClustersResponse, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.ListClusters[0:len(c.CallOptions.ListClusters):len(c.CallOptions.ListClusters)], opts...)
	var resp *containerpb.ListClustersResponse
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.ListClusters(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// GetCluster gets the details of a specific cluster.
func (c *ClusterManagerClient) GetCluster(ctx context.Context, req *containerpb.GetClusterRequest, opts ...gax.CallOption) (*containerpb.Cluster, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.GetCluster[0:len(c.CallOptions.GetCluster):len(c.CallOptions.GetCluster)], opts...)
	var resp *containerpb.Cluster
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.GetCluster(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// CreateCluster creates a cluster, consisting of the specified number and type of Google
// Compute Engine instances.
//
// By default, the cluster is created in the project's
// default network (at /compute/docs/networks-and-firewalls#networks).
//
// One firewall is added for the cluster. After cluster creation,
// the cluster creates routes for each node to allow the containers
// on that node to communicate with all other instances in the
// cluster.
//
// Finally, an entry is added to the project's global metadata indicating
// which CIDR range is being used by the cluster.
func (c *ClusterManagerClient) CreateCluster(ctx context.Context, req *containerpb.CreateClusterRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.CreateCluster[0:len(c.CallOptions.CreateCluster):len(c.CallOptions.CreateCluster)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.CreateCluster(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// UpdateCluster updates the settings of a specific cluster.
func (c *ClusterManagerClient) UpdateCluster(ctx context.Context, req *containerpb.UpdateClusterRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.UpdateCluster[0:len(c.CallOptions.UpdateCluster):len(c.CallOptions.UpdateCluster)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.UpdateCluster(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// UpdateNodePool updates the version and/or image type of a specific node pool.
func (c *ClusterManagerClient) UpdateNodePool(ctx context.Context, req *containerpb.UpdateNodePoolRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.UpdateNodePool[0:len(c.CallOptions.UpdateNodePool):len(c.CallOptions.UpdateNodePool)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.UpdateNodePool(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetNodePoolAutoscaling sets the autoscaling settings of a specific node pool.
func (c *ClusterManagerClient) SetNodePoolAutoscaling(ctx context.Context, req *containerpb.SetNodePoolAutoscalingRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetNodePoolAutoscaling[0:len(c.CallOptions.SetNodePoolAutoscaling):len(c.CallOptions.SetNodePoolAutoscaling)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetNodePoolAutoscaling(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetLoggingService sets the logging service of a specific cluster.
func (c *ClusterManagerClient) SetLoggingService(ctx context.Context, req *containerpb.SetLoggingServiceRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetLoggingService[0:len(c.CallOptions.SetLoggingService):len(c.CallOptions.SetLoggingService)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetLoggingService(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetMonitoringService sets the monitoring service of a specific cluster.
func (c *ClusterManagerClient) SetMonitoringService(ctx context.Context, req *containerpb.SetMonitoringServiceRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetMonitoringService[0:len(c.CallOptions.SetMonitoringService):len(c.CallOptions.SetMonitoringService)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetMonitoringService(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetAddonsConfig sets the addons of a specific cluster.
func (c *ClusterManagerClient) SetAddonsConfig(ctx context.Context, req *containerpb.SetAddonsConfigRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetAddonsConfig[0:len(c.CallOptions.SetAddonsConfig):len(c.CallOptions.SetAddonsConfig)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetAddonsConfig(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetLocations sets the locations of a specific cluster.
func (c *ClusterManagerClient) SetLocations(ctx context.Context, req *containerpb.SetLocationsRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetLocations[0:len(c.CallOptions.SetLocations):len(c.CallOptions.SetLocations)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetLocations(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// UpdateMaster updates the master of a specific cluster.
func (c *ClusterManagerClient) UpdateMaster(ctx context.Context, req *containerpb.UpdateMasterRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.UpdateMaster[0:len(c.CallOptions.UpdateMaster):len(c.CallOptions.UpdateMaster)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.UpdateMaster(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetMasterAuth used to set master auth materials. Currently supports :-
// Changing the admin password of a specific cluster.
// This can be either via password generation or explicitly set the password.
func (c *ClusterManagerClient) SetMasterAuth(ctx context.Context, req *containerpb.SetMasterAuthRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetMasterAuth[0:len(c.CallOptions.SetMasterAuth):len(c.CallOptions.SetMasterAuth)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetMasterAuth(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// DeleteCluster deletes the cluster, including the Kubernetes endpoint and all worker
// nodes.
//
// Firewalls and routes that were configured during cluster creation
// are also deleted.
//
// Other Google Compute Engine resources that might be in use by the cluster
// (e.g. load balancer resources) will not be deleted if they weren't present
// at the initial create time.
func (c *ClusterManagerClient) DeleteCluster(ctx context.Context, req *containerpb.DeleteClusterRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.DeleteCluster[0:len(c.CallOptions.DeleteCluster):len(c.CallOptions.DeleteCluster)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.DeleteCluster(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// ListOperations lists all operations in a project in a specific zone or all zones.
func (c *ClusterManagerClient) ListOperations(ctx context.Context, req *containerpb.ListOperationsRequest, opts ...gax.CallOption) (*containerpb.ListOperationsResponse, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.ListOperations[0:len(c.CallOptions.ListOperations):len(c.CallOptions.ListOperations)], opts...)
	var resp *containerpb.ListOperationsResponse
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.ListOperations(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// GetOperation gets the specified operation.
func (c *ClusterManagerClient) GetOperation(ctx context.Context, req *containerpb.GetOperationRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.GetOperation[0:len(c.CallOptions.GetOperation):len(c.CallOptions.GetOperation)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.GetOperation(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// CancelOperation cancels the specified operation.
func (c *ClusterManagerClient) CancelOperation(ctx context.Context, req *containerpb.CancelOperationRequest, opts ...gax.CallOption) error {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.CancelOperation[0:len(c.CallOptions.CancelOperation):len(c.CallOptions.CancelOperation)], opts...)
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		_, err = c.clusterManagerClient.CancelOperation(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	return err
}

// GetServerConfig returns configuration info about the Container Engine service.
func (c *ClusterManagerClient) GetServerConfig(ctx context.Context, req *containerpb.GetServerConfigRequest, opts ...gax.CallOption) (*containerpb.ServerConfig, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.GetServerConfig[0:len(c.CallOptions.GetServerConfig):len(c.CallOptions.GetServerConfig)], opts...)
	var resp *containerpb.ServerConfig
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.GetServerConfig(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// ListNodePools lists the node pools for a cluster.
func (c *ClusterManagerClient) ListNodePools(ctx context.Context, req *containerpb.ListNodePoolsRequest, opts ...gax.CallOption) (*containerpb.ListNodePoolsResponse, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.ListNodePools[0:len(c.CallOptions.ListNodePools):len(c.CallOptions.ListNodePools)], opts...)
	var resp *containerpb.ListNodePoolsResponse
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.ListNodePools(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// GetNodePool retrieves the node pool requested.
func (c *ClusterManagerClient) GetNodePool(ctx context.Context, req *containerpb.GetNodePoolRequest, opts ...gax.CallOption) (*containerpb.NodePool, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.GetNodePool[0:len(c.CallOptions.GetNodePool):len(c.CallOptions.GetNodePool)], opts...)
	var resp *containerpb.NodePool
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.GetNodePool(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// CreateNodePool creates a node pool for a cluster.
func (c *ClusterManagerClient) CreateNodePool(ctx context.Context, req *containerpb.CreateNodePoolRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.CreateNodePool[0:len(c.CallOptions.CreateNodePool):len(c.CallOptions.CreateNodePool)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.CreateNodePool(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// DeleteNodePool deletes a node pool from a cluster.
func (c *ClusterManagerClient) DeleteNodePool(ctx context.Context, req *containerpb.DeleteNodePoolRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.DeleteNodePool[0:len(c.CallOptions.DeleteNodePool):len(c.CallOptions.DeleteNodePool)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.DeleteNodePool(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// RollbackNodePoolUpgrade roll back the previously Aborted or Failed NodePool upgrade.
// This will be an no-op if the last upgrade successfully completed.
func (c *ClusterManagerClient) RollbackNodePoolUpgrade(ctx context.Context, req *containerpb.RollbackNodePoolUpgradeRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.RollbackNodePoolUpgrade[0:len(c.CallOptions.RollbackNodePoolUpgrade):len(c.CallOptions.RollbackNodePoolUpgrade)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.RollbackNodePoolUpgrade(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetNodePoolManagement sets the NodeManagement options for a node pool.
func (c *ClusterManagerClient) SetNodePoolManagement(ctx context.Context, req *containerpb.SetNodePoolManagementRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetNodePoolManagement[0:len(c.CallOptions.SetNodePoolManagement):len(c.CallOptions.SetNodePoolManagement)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetNodePoolManagement(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetLabels sets labels on a cluster.
func (c *ClusterManagerClient) SetLabels(ctx context.Context, req *containerpb.SetLabelsRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetLabels[0:len(c.CallOptions.SetLabels):len(c.CallOptions.SetLabels)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetLabels(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetLegacyAbac enables or disables the ABAC authorization mechanism on a cluster.
func (c *ClusterManagerClient) SetLegacyAbac(ctx context.Context, req *containerpb.SetLegacyAbacRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetLegacyAbac[0:len(c.CallOptions.SetLegacyAbac):len(c.CallOptions.SetLegacyAbac)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetLegacyAbac(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// StartIPRotation start master IP rotation.
func (c *ClusterManagerClient) StartIPRotation(ctx context.Context, req *containerpb.StartIPRotationRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.StartIPRotation[0:len(c.CallOptions.StartIPRotation):len(c.CallOptions.StartIPRotation)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.StartIPRotation(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// CompleteIPRotation completes master IP rotation.
func (c *ClusterManagerClient) CompleteIPRotation(ctx context.Context, req *containerpb.CompleteIPRotationRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.CompleteIPRotation[0:len(c.CallOptions.CompleteIPRotation):len(c.CallOptions.CompleteIPRotation)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.CompleteIPRotation(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetNodePoolSize sets the size of a specific node pool.
func (c *ClusterManagerClient) SetNodePoolSize(ctx context.Context, req *containerpb.SetNodePoolSizeRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetNodePoolSize[0:len(c.CallOptions.SetNodePoolSize):len(c.CallOptions.SetNodePoolSize)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetNodePoolSize(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetNetworkPolicy enables/Disables Network Policy for a cluster.
func (c *ClusterManagerClient) SetNetworkPolicy(ctx context.Context, req *containerpb.SetNetworkPolicyRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetNetworkPolicy[0:len(c.CallOptions.SetNetworkPolicy):len(c.CallOptions.SetNetworkPolicy)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetNetworkPolicy(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SetMaintenancePolicy sets the maintenance policy for a cluster.
func (c *ClusterManagerClient) SetMaintenancePolicy(ctx context.Context, req *containerpb.SetMaintenancePolicyRequest, opts ...gax.CallOption) (*containerpb.Operation, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.SetMaintenancePolicy[0:len(c.CallOptions.SetMaintenancePolicy):len(c.CallOptions.SetMaintenancePolicy)], opts...)
	var resp *containerpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.clusterManagerClient.SetMaintenancePolicy(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}
