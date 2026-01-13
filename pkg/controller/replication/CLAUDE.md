# Package: replication

Provides the ReplicationController manager as a wrapper around ReplicaSetController.

## Key Types

- **ReplicationManager**: Embeds ReplicaSetController and treats ReplicationController as the older API version of ReplicaSet.

## Key Functions

- **NewReplicationManager**: Creates a ReplicaSetController configured for ReplicationController resources using adapter informers.

## Design Patterns

- Uses adapters to convert ReplicationController informers to ReplicaSet informers.
- Shares all logic with ReplicaSetController.
- Exists for backward compatibility with the ReplicationController API.
- Single point of configuration difference: uses ReplicationController type instead of ReplicaSet.
