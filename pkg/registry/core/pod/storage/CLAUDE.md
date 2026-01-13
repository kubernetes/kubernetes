# Package: storage

## Purpose
Provides comprehensive REST storage implementation for Pods including all subresources: binding, eviction, status, ephemeralContainers, resize, log, proxy, exec, attach, and port-forward.

## Key Types

- **PodStorage**: Container holding all Pod-related REST handlers.
- **REST**: Main storage implementing Redirector, ShortNamesProvider, CategoriesProvider.
- **BindingREST**: Handles pod-to-node binding with preconditions.
- **LegacyBindingREST**: Legacy binding endpoint wrapper.
- **EvictionREST**: Handles pod eviction with PDB integration.
- **StatusREST**: Status subresource updates.
- **EphemeralContainersREST**: Ephemeral container updates.
- **ResizeREST**: Container resource resize updates.

## Key Functions

- **NewStorage()**: Creates PodStorage with all subresources configured. Accepts kubelet connection info, proxy transport, PDB client, and authorizer.

- **BindingREST.Create()**: Binds pod to node with preconditions, sets annotations/labels, clears scheduling gates, updates PodScheduled condition.

- **EvictionREST.Create()**: Complex eviction logic:
  - Checks if PDB can be ignored (terminal/pending pods)
  - Validates against PodDisruptionBudgets
  - Handles unhealthy pod eviction policies (AlwaysAllow, IfHealthyBudget)
  - Decrements PDB disruptions allowed
  - Adds DisruptionTarget condition before deletion

- **ShortNames()**: Returns `["po"]` for kubectl.
- **Categories()**: Returns `["all"]` for kubectl get all.

## Design Notes

- Implements `rest.Redirector` for pod resource location.
- BindingREST preserves UID/ResourceVersion preconditions from request.
- EvictionREST integrates with PDB controller via policy client.
- MaxDisruptedPodSize (2000) limits eviction when PDB controller is behind.
- Feature-gated: ClearingNominatedNodeNameAfterBinding, PodTopologyLabelsAdmission.
- All subresources share underlying store.
