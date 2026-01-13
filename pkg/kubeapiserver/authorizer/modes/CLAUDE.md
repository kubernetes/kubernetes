# Package: modes

## Purpose
This package defines the available authorization modes for the Kubernetes API server. It provides constants and validation for the supported authorization mechanisms.

## Constants

- **ModeAlwaysAllow**: Authorizes all requests (for testing/development)
- **ModeAlwaysDeny**: Denies all requests (for testing)
- **ModeABAC**: Attribute-Based Access Control authorization
- **ModeWebhook**: External webhook authorization
- **ModeRBAC**: Role-Based Access Control authorization
- **ModeNode**: Node-specific authorization for kubelets

## Key Variables

- **AuthorizationModeChoices**: List of all valid authorization mode strings

## Key Functions

- **IsValidAuthorizationMode()**: Validates if a mode string is a supported authorization mode

## Design Notes

- Mode constants are used throughout the codebase for authorization configuration
- RBAC and Node are the most commonly used modes in production
- AlwaysAllow and AlwaysDeny are primarily for testing
- ABAC is deprecated in favor of RBAC but still supported
