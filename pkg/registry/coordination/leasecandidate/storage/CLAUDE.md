# Package: storage

## Purpose
Provides etcd-backed REST storage for LeaseCandidate resources in the coordination.k8s.io API group.

## Key Types

- **REST**: Embeds `genericregistry.Store` to provide full REST semantics for LeaseCandidate resources.

## Key Functions

- **NewREST(optsGetter generic.RESTOptionsGetter)**: Creates and configures a new REST storage instance for leasecandidates.

## Configuration

The Store is configured with:
- **NewFunc/NewListFunc**: Factory functions for LeaseCandidate and LeaseCandidateList objects.
- **DefaultQualifiedResource**: "leasecandidates" (plural form).
- **SingularQualifiedResource**: "leasecandidate" (singular form).
- **CreateStrategy/UpdateStrategy/DeleteStrategy**: Uses the leasecandidate.Strategy from the parent package.
- **TableConvertor**: Uses internal printers for kubectl table output.
- **AttrFunc**: Uses leasecandidate.GetAttrs for field selection support.

## Design Notes

- Follows the standard Kubernetes registry storage pattern.
- Leverages the generic registry framework for CRUD operations against etcd.
- Supports all standard REST verbs (GET, LIST, CREATE, UPDATE, DELETE, WATCH).
