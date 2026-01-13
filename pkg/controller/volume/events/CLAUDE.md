# Package: events

## Purpose
Defines standard event reason constants used by volume controllers when recording Kubernetes events.

## Key Constants

### Binding Events
- **FailedBinding**: Volume binding operation failed.
- **VolumeMismatch**: Volume does not match expected specifications.

### Recycle Events
- **VolumeFailedRecycle**: Volume recycling operation failed.
- **VolumeRecycled**: Volume was successfully recycled.
- **RecyclerPod**: Event related to recycler pod.

### Delete Events
- **VolumeDelete**: Volume deletion succeeded.
- **VolumeFailedDelete**: Volume deletion failed.

### Provisioning Events
- **ExternalProvisioning**: Volume is being provisioned by external provisioner.
- **ProvisioningFailed**: Provisioning operation failed.
- **ProvisioningCleanupFailed**: Cleanup after failed provisioning failed.
- **ProvisioningSucceeded**: Provisioning completed successfully.

### Scheduling Events
- **WaitForFirstConsumer**: Waiting for first consumer before binding.
- **WaitForPodScheduled**: Waiting for pod to be scheduled.

### Expansion Events
- **ExternalExpanding**: Volume is being expanded by external resizer.

## Design Notes

- Used across multiple volume controllers for consistent event reasons.
- Events are recorded with these reasons via record.EventRecorder.
