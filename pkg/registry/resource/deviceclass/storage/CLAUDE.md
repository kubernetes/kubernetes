# Package: storage

## Purpose
Provides REST storage implementation for DeviceClass objects, enabling CRUD operations against the backend storage (etcd).

## Key Types

- **REST**: Wraps genericregistry.Store to implement RESTStorage for DeviceClass.

## Key Functions

- **NewREST(optsGetter)**: Creates and configures a REST storage object for DeviceClass with:
  - NewFunc/NewListFunc for object instantiation
  - Qualified resource names ("deviceclasses"/"deviceclass")
  - Uses deviceclass.Strategy for Create/Update/Delete operations
  - ReturnDeletedObject enabled
  - TableConvertor for kubectl output formatting
