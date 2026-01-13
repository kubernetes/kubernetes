# Package: serviceaccount

## Purpose
Implements service account token generation, validation, and management for Kubernetes authentication.

## Key Types
- `TokenGenerator` - Interface for generating JWT tokens for service accounts
- `Validator` - Generic interface for validating JWT claims
- `ServiceAccountTokenGetter` - Interface for retrieving service accounts, pods, secrets, and nodes
- `PublicKeysGetter` - Interface for retrieving public keys for token validation
- `jwtTokenAuthenticator` - Authenticator implementation for JWT tokens

## Key Functions
- `JWTTokenGenerator()` - Creates a token generator from a private key (RSA/ECDSA)
- `JWTTokenAuthenticator()` - Creates an authenticator for validating JWT tokens
- `Claims()` - Generates standard and private claims for bound service account tokens
- `LegacyClaims()` - Generates claims for legacy secret-based tokens
- `NewValidator()` - Creates validator for bound tokens
- `NewLegacyValidator()` - Creates validator for legacy tokens

## Token Types
- **Bound tokens**: Time-limited, bound to pod/secret/node, with JTI
- **Legacy tokens**: Secret-based, tracked with last-used labels

## Design Patterns
- Supports both RSA and ECDSA signing algorithms
- Generic type parameters for private claims validation
- OIDC-compatible token format with discovery endpoints
- Key rotation support via PublicKeysGetter interface
