# Package: rest

Provides the REST storage provider for the `flowcontrol.apiserver.k8s.io` API group with bootstrap configuration management.

## Key Types

- **RESTStorageProvider**: Implements storage provider and PostStartHookProvider for flowcontrol resources.
- **bootstrapConfigurationEnsurer**: Manages APF bootstrap configuration lifecycle.

## Key Functions

- **NewRESTStorage**: Creates APIGroupInfo with FlowSchema and PriorityLevelConfiguration storage.
- **PostStartHook**: Returns hook that initializes and maintains bootstrap APF configurations.
- **ensure**: Orchestrates suggested, mandatory, and cleanup operations for APF objects.
- **ensureSuggestedConfiguration / ensureMandatoryConfiguration**: Apply respective maintenance strategies.
- **removeDanglingBootstrapConfiguration**: Cleans up obsolete auto-managed objects.

## Design Notes

- Implements `PostStartHookProvider` to run bootstrap configuration after API server starts.
- Bootstrap runs with 5-minute initial timeout, then reconciles every minute.
- Two-phase bootstrap: suggested configs first (user-editable), then mandatory (enforced).
- Uses informer caches for efficient reads, client for writes.
- Gracefully handles concurrent modifications and transient errors with retries.
