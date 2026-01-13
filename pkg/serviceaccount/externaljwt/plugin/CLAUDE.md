# Package: plugin

## Purpose
Implements the external JWT signer plugin that delegates token signing to an external service via gRPC, enabling hardware security modules (HSMs) or external key management.

## Key Types
- `Plugin` - Main plugin struct implementing TokenGenerator interface
- `keyCache` - Caches public keys from external signer with automatic refresh
- `VerificationKeys` - Holds cached keys with timestamps and refresh hints

## Key Functions
- `New()` - Creates and initializes the external signer plugin with gRPC connection
- `GenerateToken()` - Signs a token by calling the external signer
- `GetPublicKeys()` - Returns cached public keys for token validation
- `GetServiceMetadata()` - Retrieves metadata from external signer (max token lifetime, etc.)

## Key Features
- Connects to external signer via Unix socket
- Automatic key cache synchronization with configurable timeout
- Validates JWT headers returned by external signer
- Supports excluding keys from OIDC discovery
- Notifies listeners when keys change

## Design Patterns
- Uses singleflight for deduplicating concurrent key syncs
- Atomic pointer for lock-free key cache reads
- gRPC with WaitForReady for connection reliability
- Implements serviceaccount.PublicKeysGetter interface
