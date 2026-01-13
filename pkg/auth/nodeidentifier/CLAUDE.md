# Package: nodeidentifier

## Purpose
Defines the interface for determining node identity from user authentication information.

## Key Interfaces
- `NodeIdentifier`: Interface with single method `NodeIdentity(user.Info) (nodeName string, isNode bool)`

## Key Concepts
- `nodeName`: The name of the Node API object associated with the user, may be empty if undetermined
- `isNode`: Boolean indicating whether the user.Info represents an identity issued to a node

## Design Notes
- This interface abstracts how node identities are determined from authenticated users
- Used by the node authorizer and other components that need to identify node requests
- Implementations typically check for system:node: username prefix and extract the node name
- Enables separation between authentication (who is this?) and node identification (which node?)
