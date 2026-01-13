# Package: scheduler

## Purpose
Implements the Kubernetes scheduler, which watches for unscheduled pods and assigns them to nodes based on resource requirements, constraints, and policies.

## Key Types

- **Scheduler**: Main scheduler struct containing cache, queue, profiles, and scheduling logic.
- **ScheduleResult**: Result of scheduling a pod, containing suggested host and evaluation stats.
- **schedulerOptions**: Configuration options for creating a scheduler instance.
- **Option**: Functional option type for configuring scheduler.
- **FailureHandlerFn**: Callback type for handling scheduling failures.

## Key Functions

- **New(ctx, client, informerFactory, dynInformerFactory, recorderFactory, opts...)**: Creates a new Scheduler instance with all components initialized.
- **Scheduler.Run(ctx)**: Starts the scheduling loop and blocks until context is done.
- **NewInformerFactory(cs, resyncPeriod)**: Creates SharedInformerFactory with scheduler-specific pod informer.
- **buildExtenders(logger, extenders, profiles)**: Builds scheduler extenders from configuration.
- **buildQueueingHintMap(ctx, es)**: Builds queueing hints for scheduling queue optimization.

## Key Components

- **Cache**: Stores node and pod state for scheduling decisions.
- **SchedulingQueue**: Priority queue for pending pods with backoff support.
- **Profiles**: Named scheduling profiles with different plugin configurations.
- **Extenders**: External HTTP endpoints for custom scheduling logic.
- **APIDispatcher**: Async API call dispatcher (feature-gated).
- **WorkloadManager**: Gang scheduling support (feature-gated).

## Configuration Options

- **WithProfiles**: Set scheduling profiles.
- **WithParallelism**: Set algorithm parallelism (default 16).
- **WithPercentageOfNodesToScore**: Limit node scoring for large clusters.
- **WithExtenders**: Configure external extenders.
- **WithFrameworkOutOfTreeRegistry**: Add out-of-tree plugins.

## Design Notes

- Uses informer-based caching for efficient state management.
- Supports multiple scheduling profiles for different workload types.
- Plugin-based architecture via the scheduling framework.
- Optimized for large clusters with percentage-based node scoring.
