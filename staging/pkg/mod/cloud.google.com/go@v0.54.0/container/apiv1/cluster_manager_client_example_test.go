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

package container_test

import (
	"context"

	container "cloud.google.com/go/container/apiv1"
	containerpb "google.golang.org/genproto/googleapis/container/v1"
)

func ExampleNewClusterManagerClient() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use client.
	_ = c
}

func ExampleClusterManagerClient_ListClusters() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.ListClustersRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.ListClusters(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_GetCluster() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.GetClusterRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.GetCluster(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_CreateCluster() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.CreateClusterRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.CreateCluster(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_UpdateCluster() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.UpdateClusterRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.UpdateCluster(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_UpdateNodePool() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.UpdateNodePoolRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.UpdateNodePool(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetNodePoolAutoscaling() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetNodePoolAutoscalingRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetNodePoolAutoscaling(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetLoggingService() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetLoggingServiceRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetLoggingService(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetMonitoringService() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetMonitoringServiceRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetMonitoringService(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetAddonsConfig() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetAddonsConfigRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetAddonsConfig(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetLocations() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetLocationsRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetLocations(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_UpdateMaster() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.UpdateMasterRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.UpdateMaster(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetMasterAuth() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetMasterAuthRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetMasterAuth(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_DeleteCluster() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.DeleteClusterRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.DeleteCluster(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_ListOperations() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.ListOperationsRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.ListOperations(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_GetOperation() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.GetOperationRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.GetOperation(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_CancelOperation() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.CancelOperationRequest{
		// TODO: Fill request struct fields.
	}
	err = c.CancelOperation(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleClusterManagerClient_GetServerConfig() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.GetServerConfigRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.GetServerConfig(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_ListNodePools() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.ListNodePoolsRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.ListNodePools(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_GetNodePool() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.GetNodePoolRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.GetNodePool(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_CreateNodePool() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.CreateNodePoolRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.CreateNodePool(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_DeleteNodePool() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.DeleteNodePoolRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.DeleteNodePool(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_RollbackNodePoolUpgrade() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.RollbackNodePoolUpgradeRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.RollbackNodePoolUpgrade(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetNodePoolManagement() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetNodePoolManagementRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetNodePoolManagement(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetLabels() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetLabelsRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetLabels(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetLegacyAbac() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetLegacyAbacRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetLegacyAbac(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_StartIPRotation() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.StartIPRotationRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.StartIPRotation(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_CompleteIPRotation() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.CompleteIPRotationRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.CompleteIPRotation(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetNodePoolSize() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetNodePoolSizeRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetNodePoolSize(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetNetworkPolicy() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetNetworkPolicyRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetNetworkPolicy(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClusterManagerClient_SetMaintenancePolicy() {
	ctx := context.Background()
	c, err := container.NewClusterManagerClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &containerpb.SetMaintenancePolicyRequest{
		// TODO: Fill request struct fields.
	}
	resp, err := c.SetMaintenancePolicy(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}
