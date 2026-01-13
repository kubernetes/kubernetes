# Package: validation

## Purpose
Provides validation logic for authentication API types, specifically TokenRequest validation.

## Key Constants
- `MinTokenAgeSec`: 600 (10 minutes) - minimum allowed token expiration duration

## Key Functions
- `ValidateTokenRequest(tr *TokenRequest)`: Validates a TokenRequest object

## Validation Rules
- Token expiration must be at least 10 minutes (600 seconds)
- Token expiration must not exceed 2^32 seconds
- Validates ExpirationSeconds field in the TokenRequestSpec

## Design Notes
- Simple validation focused on TokenRequest expiration bounds
- TokenReview validation is minimal as it's a status-returning API
- SelfSubjectReview requires no spec validation (only returns status)
