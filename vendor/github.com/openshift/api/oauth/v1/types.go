package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthAccessToken describes an OAuth access token.
// The name of a token must be prefixed with a `sha256~` string, must not contain "/" or "%" characters and must be at
// least 32 characters long.
//
// The name of the token is constructed from the actual token by sha256-hashing it and using URL-safe unpadded
// base64-encoding (as described in RFC4648) on the hashed result.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthAccessToken struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// clientName references the client that created this token.
	ClientName string `json:"clientName,omitempty" protobuf:"bytes,2,opt,name=clientName"`

	// expiresIn is the seconds from CreationTime before this token expires.
	ExpiresIn int64 `json:"expiresIn,omitempty" protobuf:"varint,3,opt,name=expiresIn"`

	// scopes is an array of the requested scopes.
	Scopes []string `json:"scopes,omitempty" protobuf:"bytes,4,rep,name=scopes"`

	// redirectURI is the redirection associated with the token.
	RedirectURI string `json:"redirectURI,omitempty" protobuf:"bytes,5,opt,name=redirectURI"`

	// userName is the user name associated with this token
	UserName string `json:"userName,omitempty" protobuf:"bytes,6,opt,name=userName"`

	// userUID is the unique UID associated with this token
	UserUID string `json:"userUID,omitempty" protobuf:"bytes,7,opt,name=userUID"`

	// authorizeToken contains the token that authorized this token
	AuthorizeToken string `json:"authorizeToken,omitempty" protobuf:"bytes,8,opt,name=authorizeToken"`

	// refreshToken is the value by which this token can be renewed. Can be blank.
	RefreshToken string `json:"refreshToken,omitempty" protobuf:"bytes,9,opt,name=refreshToken"`

	// inactivityTimeoutSeconds is the value in seconds, from the
	// CreationTimestamp, after which this token can no longer be used.
	// The value is automatically incremented when the token is used.
	InactivityTimeoutSeconds int32 `json:"inactivityTimeoutSeconds,omitempty" protobuf:"varint,10,opt,name=inactivityTimeoutSeconds"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthAuthorizeToken describes an OAuth authorization token
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthAuthorizeToken struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// clientName references the client that created this token.
	ClientName string `json:"clientName,omitempty" protobuf:"bytes,2,opt,name=clientName"`

	// expiresIn is the seconds from CreationTime before this token expires.
	ExpiresIn int64 `json:"expiresIn,omitempty" protobuf:"varint,3,opt,name=expiresIn"`

	// scopes is an array of the requested scopes.
	Scopes []string `json:"scopes,omitempty" protobuf:"bytes,4,rep,name=scopes"`

	// redirectURI is the redirection associated with the token.
	RedirectURI string `json:"redirectURI,omitempty" protobuf:"bytes,5,opt,name=redirectURI"`

	// state data from request
	State string `json:"state,omitempty" protobuf:"bytes,6,opt,name=state"`

	// userName is the user name associated with this token
	UserName string `json:"userName,omitempty" protobuf:"bytes,7,opt,name=userName"`

	// userUID is the unique UID associated with this token. UserUID and UserName must both match
	// for this token to be valid.
	UserUID string `json:"userUID,omitempty" protobuf:"bytes,8,opt,name=userUID"`

	// codeChallenge is the optional code_challenge associated with this authorization code, as described in rfc7636
	CodeChallenge string `json:"codeChallenge,omitempty" protobuf:"bytes,9,opt,name=codeChallenge"`

	// codeChallengeMethod is the optional code_challenge_method associated with this authorization code, as described in rfc7636
	CodeChallengeMethod string `json:"codeChallengeMethod,omitempty" protobuf:"bytes,10,opt,name=codeChallengeMethod"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthClient describes an OAuth client
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthClient struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// secret is the unique secret associated with a client
	Secret string `json:"secret,omitempty" protobuf:"bytes,2,opt,name=secret"`

	// additionalSecrets holds other secrets that may be used to identify the client.  This is useful for rotation
	// and for service account token validation
	AdditionalSecrets []string `json:"additionalSecrets,omitempty" protobuf:"bytes,3,rep,name=additionalSecrets"`

	// respondWithChallenges indicates whether the client wants authentication needed responses made in the form of challenges instead of redirects
	RespondWithChallenges bool `json:"respondWithChallenges,omitempty" protobuf:"varint,4,opt,name=respondWithChallenges"`

	// redirectURIs is the valid redirection URIs associated with a client
	// +patchStrategy=merge
	RedirectURIs []string `json:"redirectURIs,omitempty" patchStrategy:"merge" protobuf:"bytes,5,rep,name=redirectURIs"`

	// grantMethod is a required field which determines how to handle grants for this client.
	// Valid grant handling methods are:
	//  - auto:   always approves grant requests, useful for trusted clients
	//  - prompt: prompts the end user for approval of grant requests, useful for third-party clients
	GrantMethod GrantHandlerType `json:"grantMethod,omitempty" protobuf:"bytes,6,opt,name=grantMethod,casttype=GrantHandlerType"`

	// scopeRestrictions describes which scopes this client can request.  Each requested scope
	// is checked against each restriction.  If any restriction matches, then the scope is allowed.
	// If no restriction matches, then the scope is denied.
	ScopeRestrictions []ScopeRestriction `json:"scopeRestrictions,omitempty" protobuf:"bytes,7,rep,name=scopeRestrictions"`

	// accessTokenMaxAgeSeconds overrides the default access token max age for tokens granted to this client.
	// 0 means no expiration.
	AccessTokenMaxAgeSeconds *int32 `json:"accessTokenMaxAgeSeconds,omitempty" protobuf:"varint,8,opt,name=accessTokenMaxAgeSeconds"`

	// accessTokenInactivityTimeoutSeconds overrides the default token
	// inactivity timeout for tokens granted to this client.
	// The value represents the maximum amount of time that can occur between
	// consecutive uses of the token. Tokens become invalid if they are not
	// used within this temporal window. The user will need to acquire a new
	// token to regain access once a token times out.
	// This value needs to be set only if the default set in configuration is
	// not appropriate for this client. Valid values are:
	// - 0: Tokens for this client never time out
	// - X: Tokens time out if there is no activity for X seconds
	// The current minimum allowed value for X is 300 (5 minutes)
	//
	// WARNING: existing tokens' timeout will not be affected (lowered) by changing this value
	AccessTokenInactivityTimeoutSeconds *int32 `json:"accessTokenInactivityTimeoutSeconds,omitempty" protobuf:"varint,9,opt,name=accessTokenInactivityTimeoutSeconds"`
}

type GrantHandlerType string

const (
	// GrantHandlerAuto auto-approves client authorization grant requests
	GrantHandlerAuto GrantHandlerType = "auto"
	// GrantHandlerPrompt prompts the user to approve new client authorization grant requests
	GrantHandlerPrompt GrantHandlerType = "prompt"
	// GrantHandlerDeny auto-denies client authorization grant requests
	GrantHandlerDeny GrantHandlerType = "deny"
)

// ScopeRestriction describe one restriction on scopes.  Exactly one option must be non-nil.
type ScopeRestriction struct {
	// ExactValues means the scope has to match a particular set of strings exactly
	ExactValues []string `json:"literals,omitempty" protobuf:"bytes,1,rep,name=literals"`

	// clusterRole describes a set of restrictions for cluster role scoping.
	ClusterRole *ClusterRoleScopeRestriction `json:"clusterRole,omitempty" protobuf:"bytes,2,opt,name=clusterRole"`
}

// ClusterRoleScopeRestriction describes restrictions on cluster role scopes
type ClusterRoleScopeRestriction struct {
	// roleNames is the list of cluster roles that can referenced.  * means anything
	RoleNames []string `json:"roleNames" protobuf:"bytes,1,rep,name=roleNames"`
	// namespaces is the list of namespaces that can be referenced.  * means any of them (including *)
	Namespaces []string `json:"namespaces" protobuf:"bytes,2,rep,name=namespaces"`
	// allowEscalation indicates whether you can request roles and their escalating resources
	AllowEscalation bool `json:"allowEscalation" protobuf:"varint,3,opt,name=allowEscalation"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthClientAuthorization describes an authorization created by an OAuth client
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthClientAuthorization struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// clientName references the client that created this authorization
	ClientName string `json:"clientName,omitempty" protobuf:"bytes,2,opt,name=clientName"`

	// userName is the user name that authorized this client
	UserName string `json:"userName,omitempty" protobuf:"bytes,3,opt,name=userName"`

	// userUID is the unique UID associated with this authorization. UserUID and UserName
	// must both match for this authorization to be valid.
	UserUID string `json:"userUID,omitempty" protobuf:"bytes,4,opt,name=userUID"`

	// scopes is an array of the granted scopes.
	Scopes []string `json:"scopes,omitempty" protobuf:"bytes,5,rep,name=scopes"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthAccessTokenList is a collection of OAuth access tokens
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthAccessTokenList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of OAuth access tokens
	Items []OAuthAccessToken `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthAuthorizeTokenList is a collection of OAuth authorization tokens
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthAuthorizeTokenList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of OAuth authorization tokens
	Items []OAuthAuthorizeToken `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthClientList is a collection of OAuth clients
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthClientList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of OAuth clients
	Items []OAuthClient `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthClientAuthorizationList is a collection of OAuth client authorizations
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthClientAuthorizationList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of OAuth client authorizations
	Items []OAuthClientAuthorization `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuthRedirectReference is a reference to an OAuth redirect object.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthRedirectReference struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// The reference to an redirect object in the current namespace.
	Reference RedirectReference `json:"reference,omitempty" protobuf:"bytes,2,opt,name=reference"`
}

// RedirectReference specifies the target in the current namespace that resolves into redirect URIs.  Only the 'Route' kind is currently allowed.
type RedirectReference struct {
	// The group of the target that is being referred to.
	Group string `json:"group" protobuf:"bytes,1,opt,name=group"`

	// The kind of the target that is being referred to.  Currently, only 'Route' is allowed.
	Kind string `json:"kind" protobuf:"bytes,2,opt,name=kind"`

	// The name of the target that is being referred to. e.g. name of the Route.
	Name string `json:"name" protobuf:"bytes,3,opt,name=name"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UserOAuthAccessToken is a virtual resource to mirror OAuthAccessTokens to
// the user the access token was issued for
// +openshift:compatibility-gen:level=1
type UserOAuthAccessToken OAuthAccessToken

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UserOAuthAccessTokenList is a collection of access tokens issued on behalf of
// the requesting user
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type UserOAuthAccessTokenList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Items []UserOAuthAccessToken `json:"items" protobuf:"bytes,2,rep,name=items"`
}
