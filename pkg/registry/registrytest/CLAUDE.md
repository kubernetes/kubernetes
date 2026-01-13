# Package: registrytest

## Purpose
Provides mock implementations of Registry interfaces for testing purposes. These mocks are used to test registry implementations that store Kubernetes resources like Nodes, Endpoints, Services, and Pods.

## Key Types

- **EndpointRegistry**: Mock registry for Endpoints that implements List, Get, Create, Update, Delete operations with thread-safe access via mutex.
- **NodeRegistry**: Mock registry for Nodes supporting ListNodes, CreateNode, UpdateNode, GetNode, DeleteNode, WatchNodes.
- **ServiceRegistry**: Mock registry for Services with similar CRUD operations and namespace-aware listing.

## Key Functions

- **NewEtcdStorage(t, group)**: Creates an etcd storage configuration for testing with a bogus resource.
- **NewEtcdStorageForResource(t, resource)**: Creates etcd storage for a specific GroupResource.
- **MakeNodeList(nodes, resources)**: Helper to construct api.NodeList from node names and resources.
- **AssertCategories(t, storage, expected)**: Validates that a storage provider returns expected categories.
- **AssertShortNames(t, storage, expected)**: Validates that a storage provider returns expected short names.
- **ValidateStorageStrategies(storageMap)**: Ensures all generic registry stores have Create, Update, and Delete strategies defined.

## Design Patterns

- All mock registries use mutex locks for thread-safe concurrent access.
- Registries store configurable error values (Err field) to simulate error conditions in tests.
- Mock registries track operations (Updates, DeletedID, GottenID) for test assertions.
