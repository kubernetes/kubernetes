# Package: apparmor

## Purpose
Provides utilities for validating and working with AppArmor security profiles in Kubernetes pods.

## Key Types
- `Validator` - Interface for validating AppArmor profile configurations

## Key Functions
- `ValidateHost()` - Validates AppArmor is available and profiles are loaded on the host
- `GetProfile()` - Extracts AppArmor profile from pod annotations or security context
- `ValidateProfile()` - Validates a profile string has correct format

## Supported Profile Types
- `runtime/default` - Uses container runtime's default profile
- `localhost/<profile-name>` - Uses a profile loaded on the node
- `unconfined` - Disables AppArmor for the container

## Design Patterns
- Interface-based design for testability
- Validates both annotation-based and field-based AppArmor configuration
- Checks profile availability on the host system
