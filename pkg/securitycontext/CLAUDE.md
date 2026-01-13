# Package: securitycontext

## Purpose
Provides accessor and mutator interfaces for working with pod and container security contexts in a uniform way.

## Key Types
- `PodSecurityContextAccessor` - Read-only access to pod security context fields
- `PodSecurityContextMutator` - Read-write access to pod security context fields
- `ContainerSecurityContextAccessor` - Read-only access to container security context
- `ContainerSecurityContextMutator` - Read-write access to container security context

## Key Functions
- `NewPodSecurityContextAccessor()` - Creates accessor for pod security context
- `NewPodSecurityContextMutator()` - Creates mutator for pod security context
- `NewEffectiveContainerSecurityContextAccessor()` - Creates accessor with effective values
- `NewEffectiveContainerSecurityContextMutator()` - Creates mutator with effective values

## Design Patterns
- Accessor/Mutator pattern for clean separation of read/write operations
- "Effective" accessors merge pod-level and container-level settings
- Handles nil security contexts gracefully
- Provides uniform interface regardless of whether security context is set
