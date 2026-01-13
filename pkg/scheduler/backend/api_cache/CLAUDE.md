# Package: api_cache

## Purpose
Provides caching for API calls made by the scheduler. The APICacher batches and optimizes API calls (like pod bindings and status patches) to reduce load on the API server while ensuring data consistency.

## Key Types

### APICacher
The main struct that implements the `framework.APICacher` interface. It manages pending API calls and coordinates with the APIDispatcher.

## Key Functions

- **New(dispatcher api_dispatcher.APIDispatcher)**: Creates a new APICacher with the given dispatcher
- **BindPod(binding *v1.Binding)**: Queues a pod binding operation and returns a finish channel
- **UpdatePodCondition(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo)**: Queues a pod condition update
- **WaitOnFinish(ctx context.Context, onFinish <-chan FinishStatus)**: Blocks until the API call completes
- **Get(uid types.UID) metav1.Object**: Retrieves the current (potentially updated) state of an object

## Design Pattern
- Uses a two-phase approach: calls are first cached/merged, then dispatched in batches
- Supports call merging for efficiency (e.g., multiple status patches to the same pod)
- The `onFinish` channel pattern allows callers to wait for API call completion
- Integrates with informer updates via the `Sync` method to keep cached objects consistent
