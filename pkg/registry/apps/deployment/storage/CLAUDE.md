# Package: storage

## Purpose
Provides REST storage implementation for Deployment resources with status, scale, and rollback subresources.

## Key Types

- **DeploymentStorage**: Container for all Deployment-related REST endpoints
- **REST**: Main REST storage for Deployment CRUD operations
- **StatusREST**: REST endpoint for /status subresource
- **ScaleREST**: REST endpoint for /scale subresource (autoscaling)
- **RollbackREST**: REST endpoint for /rollback subresource (deprecated)
- **scaleUpdatedObjectInfo**: Transforms deployment->scale->deployment for scale updates

## Key Functions

- **NewStorage(optsGetter)**: Creates DeploymentStorage with all subresource handlers
- **NewREST(optsGetter)**: Creates REST, StatusREST, and RollbackREST instances
- **ReplicasPathMappings()**: Returns replicas field paths for different API versions
- **ScaleREST.Get()**: Converts Deployment to Scale object
- **ScaleREST.Update()**: Updates replicas via Scale, converts back to Deployment
- **RollbackREST.Create()**: Triggers deployment rollback (deprecated API)
- **scaleFromDeployment()**: Extracts Scale subresource from Deployment

## Design Notes

- Implements ShortNamesProvider ("deploy") and CategoriesProvider ("all")
- Scale subresource supports multiple API versions (apps/v1, extensions/v1beta1, etc.)
- Rollback subresource is for deprecated extensions/v1beta1 API
- Uses managed fields handler for proper scale subresource field tracking
- Scale operations validate via autoscaling validation
