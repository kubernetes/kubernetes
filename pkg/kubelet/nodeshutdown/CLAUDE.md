# Package: nodeshutdown

Manages graceful node shutdown, terminating pods with appropriate grace periods before system shutdown.

## Key Types/Structs

- **Manager**: Interface for node shutdown management with Admit, Start, and ShutdownStatus methods.
- **Config**: Configuration including grace periods, pod priority settings, and dependencies.
- **podManager**: Terminates pods by priority class with configurable grace periods.
- **podShutdownGroup**: Groups pods by priority with associated grace period.
- **managerStub**: No-op implementation for non-Linux platforms.

## Key Functions

- `newPodManager()`: Creates pod manager with priority-based grace periods.
- `killPods()`: Terminates pods in priority order (low to high), waiting for volume unmounts.
- `groupByPriority()`: Organizes pods into shutdown groups by priority.
- `migrateConfig()`: Converts old config format (two grace periods) to priority-based format.

## Pod Termination Logic

1. Pods grouped by priority class
2. Lower priority pods terminated first
3. Each group waits up to its configured grace period
4. Volumes waited for unmount (best effort)
5. Pod status set to Failed with "Terminated" reason

## Configuration Options

- `ShutdownGracePeriodRequested`: Total shutdown time requested
- `ShutdownGracePeriodCriticalPods`: Time reserved for critical pods
- `ShutdownGracePeriodByPodPriority`: Fine-grained per-priority configuration

## Design Notes

- Implements PodAdmitHandler to reject new pods during shutdown
- Stores shutdown state for persistence across restarts
- Linux uses systemd inhibitor locks; Windows/others use stub implementation
- Pods get DisruptionTarget condition with PodReasonTerminationByKubelet
