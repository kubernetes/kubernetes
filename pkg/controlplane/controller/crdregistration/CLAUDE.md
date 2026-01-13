# Package: crdregistration

## Purpose
This controller automatically registers APIService resources for CustomResourceDefinitions (CRDs). When a CRD is created, modified, or deleted, the controller ensures corresponding APIService entries exist for auto-registration with the aggregator.

## Key Types

- **crdRegistrationController**: Watches CRDs and manages APIService registration via the AutoAPIServiceRegistration interface
- **AutoAPIServiceRegistration**: Interface for adding/removing APIServices to/from the auto-registration system

## Key Functions

- **NewCRDRegistrationController()**: Creates a controller that watches CRD informer events
- **Run()**: Starts the controller, processes initial CRD set, then starts workers
- **WaitForInitialSync()**: Blocks until the initial set of CRDs has been processed
- **handleVersionUpdate()**: For a given group/version, either adds or removes the corresponding APIService from auto-registration
- **enqueueCRD()**: Enqueues all group/versions from a CRD for processing

## APIService Configuration for CRDs

CRDs get APIServices with:
- GroupPriorityMinimum: 1000 (relatively low priority)
- VersionPriority: 100 (allows kube-like version sorting)

## Design Notes

- Processes both old and new objects on update to handle version changes
- Only registers APIServices for CRD versions that are served (version.Served == true)
- Removes APIService from auto-registration when CRD is deleted or version is no longer served
- Uses workqueue keyed by schema.GroupVersion for deduplication
