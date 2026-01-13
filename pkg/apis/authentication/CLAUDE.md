# Package: authentication

## Purpose
Defines the internal (unversioned) API types for the authentication.k8s.io API group, handling token validation, token requests, and user identity information.

## Key Types

### TokenReview
Attempts to authenticate a token to a known user.
- `TokenReviewSpec`: Contains the token to validate and optional audience list
- `TokenReviewStatus`: Returns authentication result, user info, and matched audiences

### TokenRequest
Requests a token for a given service account.
- `TokenRequestSpec`: Specifies audiences, expiration duration, and optional bound object reference
- `TokenRequestStatus`: Returns the issued token and expiration timestamp
- `BoundObjectReference`: Binds token validity to a Pod or Secret lifecycle

### SelfSubjectReview
Returns the user information that the API server has about the requesting user.
- `SelfSubjectReviewStatus`: Contains the UserInfo of the requester

### UserInfo
Holds user identity information.
- `Username`: Unique user identifier
- `UID`: Unique value identifying user across time
- `Groups`: Group memberships
- `Extra`: Additional authenticator-provided information

## Key Constants
- `ImpersonateUserHeader`: "Impersonate-User" - header for user impersonation
- `ImpersonateUIDHeader`: "Impersonate-Uid" - header for UID impersonation
- `ImpersonateGroupHeader`: "Impersonate-Group" - header for group impersonation
- `ImpersonateUserExtraHeaderPrefix`: "Impersonate-Extra-" - prefix for extra info impersonation

## Key Functions
- `AddToScheme`: Registers TokenReview, TokenRequest, and SelfSubjectReview types
- `Kind(kind string)`: Returns Group-qualified GroupKind
- `Resource(resource string)`: Returns Group-qualified GroupResource
