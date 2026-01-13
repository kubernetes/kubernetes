# Package: authority

## Purpose
Implements a certificate authority (CA) for signing certificate requests with policy-based controls.

## Key Types/Structs
- `CertificateAuthority`: Contains CA certificate, private key, and raw cert/key bytes for change detection
- `SigningPolicy`: Interface for policies that control certificate issuance
- `PermissiveSigningPolicy`: Policy with configurable TTL, usages, backdating, and short-cert handling

## Key Functions
- `Sign(crDER []byte, policy SigningPolicy) ([]byte, error)`: Signs a DER-encoded certificate request with the given policy
- `NewCAProvider(certFile, keyFile)`: Creates a CA provider that watches files for changes

## Sign Operation
1. Parse and verify the certificate request signature
2. Generate a random 128-bit serial number
3. Create certificate template from CSR (subject, SANs, extensions)
4. Apply signing policy (TTL, key usages, validity period)
5. Sign with CA private key and return DER-encoded certificate

## PermissiveSigningPolicy Fields
- `TTL`: Maximum certificate duration
- `Usages`: Allowed key usages (client auth, server auth, etc.)
- `Backdate`: Amount to backdate NotBefore (default 5 min for clock skew)
- `Short`: Threshold below which short-cert backdating applies
- `Now`: Optional time function for testing

## Design Notes
- Serial number limit is 2^128 (128 bits)
- CA provider supports dynamic reloading of cert/key files
- Policy controls what the resulting certificate can be used for
