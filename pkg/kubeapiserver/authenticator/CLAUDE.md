# Package: authenticator

## Purpose
This package configures the authentication chain for the Kubernetes API server. It creates a union authenticator from multiple authentication methods including X509 client certificates, bearer tokens, service accounts, OIDC, and webhooks.

## Key Types

- **Config**: Authentication configuration including all supported auth methods
- **jwtAuthenticatorWithCancel**: JWT authenticator wrapper with health check and cancellation
- **authenticationConfigUpdater**: Handles dynamic reloading of JWT authenticators

## Key Functions

- **Config.New()**: Creates the complete authenticator chain with OpenAPI security definitions
- **newJWTAuthenticator()**: Creates OIDC/JWT token authenticators from config
- **newServiceAccountAuthenticator()**: Creates service account token authenticator
- **newWebhookTokenAuthenticator()**: Creates webhook-based token authenticator
- **updateAuthenticationConfig()**: Dynamically updates JWT authenticators without restart

## Authentication Methods (in order)

1. Front-proxy (request header) authentication
2. X509 client certificate authentication
3. Token file authentication
4. Service account token authentication (legacy and bound)
5. Bootstrap token authentication
6. OIDC/JWT authentication
7. Webhook token authentication
8. Anonymous authentication (if enabled, as fallback)

## Design Notes

- Token authenticators are cached with configurable TTL for performance
- OIDC authenticators support dynamic configuration reloading
- Service account authenticators should be ordered before OIDC to avoid cache misses
- Anonymous auth only runs if all other authenticators fail (not on errors)
- Generates OpenAPI v2/v3 security definitions for documentation
