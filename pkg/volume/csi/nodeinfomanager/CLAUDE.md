# Package: nodeinfomanager

## Purpose
Manages CSI driver information on Kubernetes Node objects, including updating node annotations and CSINode objects with driver topology and capabilities.

## Key Types/Structs
- `nodeInfoManager` - Main manager for CSI node information
- `Interface` - Interface defining node info management operations

## Key Functions
- `NewNodeInfoManager()` - Creates a new node info manager
- `InstallCSIDriver()` - Registers a CSI driver on the node (updates CSINode CR)
- `UninstallCSIDriver()` - Removes a CSI driver from the node
- `CreateCSINode()` - Creates the CSINode custom resource for this node

## Design Patterns
- Updates CSINode custom resource with driver topology keys
- Manages node allocatable for volume limits
- Handles driver migration annotations
- Coordinates with kubelet plugin watcher for driver lifecycle
