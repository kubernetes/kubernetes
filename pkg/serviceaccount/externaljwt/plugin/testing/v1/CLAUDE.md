# Package: v1

## Purpose
Provides a mock implementation of the external JWT signer gRPC service for testing the external JWT plugin.

## Key Types
- `MockSigner` - Mock gRPC server implementing ExternalJWTSignerServer
- `KeyT` - Represents a public key with optional OIDC discovery exclusion flag

## Key Functions
- `NewMockSigner()` - Creates and starts a new mock signer on a Unix socket
- `Sign()` - Mock implementation that signs JWTs using RSA
- `FetchKeys()` - Returns configured public keys
- `Metadata()` - Returns mock metadata (max token expiration, etc.)
- `Reset()` - Regenerates signing keys and resets error state

## Test Support Features
- Configurable signing key, algorithm, and key ID
- Injectable errors for FetchKeys and Metadata calls
- Thread-safe key manipulation with condition variables
- Automatic cleanup on test completion
- Wait helpers for synchronizing with key fetches

## Design Patterns
- Implements gRPC ExternalJWTSignerServer interface
- Uses RSA-256 signing by default
- Supports testing key rotation scenarios
- Provides hooks for error injection testing
