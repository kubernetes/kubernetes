# Plugin Architecture for Authentication and Authorization

_This document describes features which are not yet implemented yet._

Different K8s cluster owners may wish to use different authentication and authorization systems.
  - enterprises may wish to integrate with existing user and group database.  They may prefer to interact with LDAP or SAML protocols.
  - companies that use IAM-as-a-service (e.g. Stormpath, Auth0, etc.) for existing software may want to use that for k8s too.
  - a hosted kubernetes offering might integrate with the existing identity and groups systems of the hosting-provider.
  - and so on.

Similarly, different cluster owners may prefer different authorization schemes
  - a small cluster may authorize all users to do any action
  - a cluster on Amazon might want to use authorization policies similar to Amazon IAM policies
  - a cluster with OpenShift on top of it might use the same role-based authorization scheme.

Some of these extensions may be contributed back to the kubernetes github project, and others may remain private.

This document defines stable interfaces for alternative authentication and authorization schemes to plug in to the core of
kubernetes.   

Not all external auth systems will be designed to work at high request rates.  High request rates are
expected as automation is built on top of the kubernetes API.  Therefore, kubernetes
will need to provide a short-term authentication mechanism and cache authorization decisions.  This is also described.

### Prerequisites
This document refers to _API Plugins_, like discussed in [#1355](https://github.com/GoogleCloudPlatform/kubernetes/pull/1355).

## Design Summary
Here are the key points of the design:
  - APIserver Architecture
    - Authentication, Group Membership, and Authorization are handled by separate interfaces in separate steps by the APIserver.
    - Cluster owners can chose from among contributed pacakges, or write their own go code to handle each of those steps.
    - A single go package might implement one, two or all three interfaces with a single type.
    - An implementation of one those interfaces might use external storage or it might add new k8s REST resources (that is, also be an API Plugin).
  - Authentication
    - An `/auth` endpoint on the apiserver generates short term random unscoped Bearer tokens which identify a user in subsequent API calls.
    - API calls must include a valid Bearer token in an `Authorization` header.
    - An IdentityProvider is go code that implements the `IdentityProvider` interface.
    - The `/auth` handler uses an IdentityProvider to redirect the user to an actual identity provider.
    - The IdentityProvider may use Oauth to communicate with the identity provider (google, facebook, github) or not (e.g. SAML).
  - Group membership
    - A GroupProvider is go code that implements the `GroupProvider` interface.
    - The `auth` endpoint checks group membership right after checking identity.  The Bearer token it generates
      attests to group membership as well as identity.
  - Authorization
    - The APIserver passes context about an API call to an AuthorizationProvider which makes an authorization
      decision.
    - An AuthorizationProvider is a go code that implements the `AuthorizationProvider` interface.
    - The AuthorizationProvider is responsible for storing any policies, permissions,  or access control lists that it
      needs to make decisions.  These might be stored as resources in the APIserver via an API plugin, or they might be
      in a separate policy database.
    - The context includes the _user_ string, and _group_ strings of a request, the HTTP method, the resource path, and
      other facts about the request.
  - User and groups
    - Users and groups have no inherent meaning to the APIserver.  They only have meaning defined by the
      IdentityProvider and GroupProvider.  That meaning may be relied on by the AuthorizationProvider to make decisions.
  - Default Implementation
    - Kubernetes will by default be configured to use trivial implementations of AuthenticationProvider, GroupProvider,  and
      AuthorizationProvider (collectively, Auth Plugins) that can authenticate a single user and authorize all actions.
    - The community can contribute Auth Plugins that integrate with a range of existing solutions.

## Detailed Design

### Authentication
The key components of k8s Authentication are:
  - an APIserver handler for `/auth/...` which gives clients Bearer Tokens to use with the `/api/...` endpoints.
  - an IdentityProvider interface which is called by the `/auth` handler to determine identity (user string).
  - an GroupProvider interface which is called by the `/auth` handler to determine group membership (slice of group name
    strings).
  - an APIserver handler for `/api/...` which checks for Bearer Tokens, maps them to user/group strings, or rejects
    unmappable/invalid requests.

#### The auth endpoint

A first time user of the APIserver needs a token to access the API.  The user can either:
  - manually go to the `/auth` endpoint, which will show a page with the Token, and then the user can store the token.
  - clients like kubecfg can automate this process.

The auth endpoint handler holds an variable which implements the Go interface called `IdentityProvider`.  
It calls this to determine the authenticated identity of the client, or what error to return.
APIserver also holds a variable which implements the `GroupProvider` interface.  This is used to lookup the group memberships of
a user string.

After successful authentication and group membership checking, the auth handler shows the user a page containing a
token which can be used for API calls.

APIserver maintains a map from tokens to user and group information, like this:
```go
type userAndGroups struct {
  expires int64
  user string
  groups []string
}
type token []bytes
type tokenMap map[token]userAndGroups
var tokens := make(tokenMap)
```

The auth handler populates the map.  A goroutine scans periodically for expired entries.

#### IdentityProvider interface
The IdentityProvider interface separates the responsibilities between core kubernetes code and contributed or private
code.  Kubernetes core owns generating the tokens used by `/api/...`, setting the lifetime of those tokens, and
presenting the token to the user.  The IdentityProvider owns identifying a user.

This is the interface:
```go
type UserOrRedirect {
  // Only one is non-empty.
  User string
  Redirect string
  }
type IdentityProvider interface {
  HandleClient(req http.Request) (userOrRedirect UserOrRedirect, err error)
  HandleProvider(req http.Request) (user string, err error)
}
```
The interface allows for two different flows:
  1. One Request flow
    - User comes to `/auth`.  
    - HandleClient is called.
    - HandleClient extracts information from the headers, query parameters, or cookies, and uses those to authenticate the user. 
    - HandleClient may RPC or make web requests to an external source to complete the authentication.
    - HandleClient returns the user string.
    - In this scenatiom HandleProvider is not expected to be called, and the implementation should always return an error.
  1. Two Request flow
    - User comes to `/auth`.
    - HandleClient is called.
    - HandleClient returns a redirect URL for the user to go to to complete Auth.
    - User completes auth at a remote site.
    - The remote site redirects user back to `/auth/provider?...`. 
    - HandleProvider is called.
    - HandleProvider may make more external requests.
    - HandleProvider returns user string.

The One Request Flow is suitable for where kubernetes and actual identity provider are on the same domain and so cookies
can be used (e.g. SSO).
The Two Request Flow is suitable for an OpenID style of authentication where the actual identity provider is on another
domain.

See the examples section for a concrete example of the Two Request Flow.

*TODO*: should there be a different path under `/auth` for each possible IdentityProvider, (e.g. `/auth/google`,
`/auth/facebook`, `/auth/ldap`, `/auth/sso`) to allow multiple concurrent IdentityProviders?

#### GroupProvider interface
Group Member checking as a step that is right after authentication and before authorization.

The GroupProvider is similar to the IdentityProvider interface.

*TODO: work out flows for group provider.*

An actual  group provider may have many groups, and may not support searching for all the groups that a user belongs to,
and may not support discovering what groups exist.  Kubernetes only need to know about groups that are referenced by
authorization policies.  Therefore, the AuthorizationProvider will have an interface to get all the groups which are
relevant to authorization.  Apiserver will provide the GroupProvider with this list.

See Examples section for example of groups handling.

#### The API handler

The `/api` handler has the following general flow:
  1. Check for bearer auth.  If not present, reply `401 Unauthorised` with `WWW-Authenticate` header is set.  
  1. Check if token is found in `tokens`.  If not reply 401 as above,
  1. Get the user and groups from this token.  These will be used in the authorization step described later.

 *TODO*: reconcile this with that @smarterclayton said:
   - The authn provider should populate WWW-Authenticate but for the majority of calls I would prefer we define that the server will only challenge Bearer but clients must be flexible to receiving other types, in which case they are allowed to treat the reauth as "non actionable". What that allows is SSL client auth and Kerb proxies to sit in front of bearer auth and offer an additional level of control.

### Authorization

All `/api/...` calls sent to the kubernetes API server are subject to authorization.

The APIserver handles authorization after authentication and group-membership.

The APIserver assembles relevant _attributes_ from the authentication step, and from the current HTTP request.  It
passes these attributes to the `Authorize` function of an `AuthorizationProvider` interface which allows or denies the action.

The APIserver will have a caching layer with configurable TTL to avoid repeated calls to `Authorize` with the same
arguments in a short time span.  As a consequence, policy changes will not take instant effect, but will take effect
within a bounded time.

### AuthorizationProvider interface
The AuthorizationProvider typically compares some or all of these attributes with corresponding fields policies or
access control lists which it stores.    Not every implementation of an AuthorizationProvider needs to consider all the
attributes.

The AuthorizationProvider interface:
```go
type AuthorizationProvider interface {
  Authorize(a Attributes) (err error)
  GetRelevantGroups() (groups []string, err error)
}
```

The Attributes and error codes are described in the following sections.

#### Attributes

These are the attributes:
```go
type Attributes struct {
  // Subject attributes
  User string       // a user identifier (e.g. "alice@example.com" or "alice.openid.example.org")
  Groups []string   // a list of groups the user belongs to, also from the IdProvider.

  // Verb
  Verb string       // GET, PUT, POST, or DELETE
  VerbModifier string  //  e.g. "Watch", "Restart", "Proxy"

  // Object attributes
  Namespace string   // the namespace of the resource being accessed
                     // (empty if the resource is a namespace or object is not namespaced)
                     // requires https://github.com/GoogleCloudPlatform/kubernetes/pull/1114

  Kind string        // the Kind of object
  Name string        // the Name of the object, per #1124 meaning of Name.
  WithCapabilities []string // e.g. Restricted capability
}
```
They are broadly divided into Subject, Verb, and Object attributes.

The Subject attributes are about the user making the call.
The user and group strings are defined by the IdentityProvider and GroupProvider.  The cluster owner is responsible for
picking a set of IdentityProvider, GroupProvider, and AuthorizationProvider implementations that have a common
understanding of these strings.

The object attributes are about the HTTP resource being created, read, modified, or otherwise operated on.
When there are multiple resource being acted on (such as listing or watching), then Object attributes may be
unspecified (empty strings). In most cases, at least Namespace should be set, and this is expected to be one of the most
important attributes for authorization.

The verb attributes are about what is being done to the resource(s).
The Verb is the HTTP method being used such as GET, POST, or PUT.  This will always be available.
Optionally, the APIserver and API Plugins can define a mapping between a request and a VerbModifier.  For example, all
requests that have "?watch" might have VerbModifier WATCH.  Other examples are RESTART and PROXY.

#### Errors and HTTP Response codes

A `nil` error from `Authorize` allows the API action to proceed.  A `NotFound` error causes the user to see  `404 Not
Found`.  A `Forbidden` causes the user to see  `403 Forbidden`.  Any other error causes a `500 Internal Error`.

The AuthorizeProvider controls whether Forbidden or NotFound is returned, but the following are suggestions of how to
determine which to show.

The HTTP responses for authentication and authorization failures should strike a balance between allowing debugging and preventing information leakage.
 - If object simply does not exist, return NotFound.
 - Else if the _subject_ has no permissions within the namespace mentioned in the _namespace_ attribute, then return
   NotFound.  This prevents leakage of names of namespaces and of objects within namespaces that the user has no need to
   know.  This protects what might be sensitive information, especially in a multi-tenant setup.
 - Else if the _subject_ does not have permission to _verb_ that _object_, then return Forbidden.  Returning forbidden
   tells the user that the object name is correct, but the permissions are wrong.

### Overall order of operations

For an `/api` call
 1. Authenticate using Token, to get user and groups.
 1. Translate request to canonical representation.
 1. Map request to VerbModifier.
 1. Compose all the attributes
 1. Call Authorize
 1. Validate the request (validation after authorization prevents info leakage and minimizes amount of code that unauthorized requests can touch)
 1. Audit Log (future work)
 1. Admission Control (future work)


### Caching

Each AuthorizationProvider implementation should implement its own caching.  The AuthorizationProvider:
  - knows which attributes matter and which do not, so it can construct a cache key which will apply to the largest set
    of subsequent Authorizations.
  - it is in the best position to know when policies change, and thus to revoke cached decisions.
  
## Examples

### Example IdentityProvider

Suppose a kubernetes cluster admin wants use github as the only identity provider for her cluster.

After reading the [Github oauth reference](https://developer.github.com/v3/oauth/), she writes a package that
defines a `GitHubIdentityProvider` type which implements the `IdentityProvider` interface, and adds a flag to that causes
the `NewIdentityProvider()` factory to return a `GitHubIdentityProvider`.

The GitHubIdentityProvider type implements the `HandleClient(req)` method like this:
  1. Return error if req is not a GET, or if it has any parameters.
  1. Return a redirect to "GET https://github.com/login/oauth/authorize"
    - Set `client_id=` to the id provided by github.
    - Set `redirect_uri=` to `http://$APISERVER/auth/provider`
    - Set `scopes=` to empty string to just know the users identity.
    - Set `state=` to random string
  1. Make a note of the progress of this authentication in a map keyed by `state` which is private to `GitHubIdentityProvider`.

The GitHubIdentityProvider implements HandleProvider as:
  1. If there is a `state` URL parameter, and `state` is found in map, then get `code` from URL (provided by github).
  1. `GET https://github.com/login/oauth/access_token` using `code`.
  1. Send access token with request for `http://api.github.com/user`
  1. Get `login` property from response body.  Return this as the user string.

The complete flow is then:
  1. User contacts `/auth` for first time
  1. The apiserver calls HandleClient for all requests to `/auth`.
  1. HandleClient returns a the url to redirected to github authorization page.
  1. User logs in if necessary and then authorizes kubernetes to access his github identity.
  1. Github redirects back to `/auth/provider?code=12345`.
  1. The apiserver calls HandleProvider for all requests to `/auth/provider`.
  1. HandleProvider extracts code query parameter.
  1. HandleProvider calls github to get access token.
  1. HandleProvider calls github with access token to get user info.
  1. HandleProvider returns the users "login" as the Kubernetes user string.
  1. APIserver generates a Token and stores username in a map keyed by the Token.
  1. User calls `/api/...` with Token
  1. APIserver lookup Token in map.
  1. User is authenticated.

### Example GroupProvider

*TODO: expand this*

Using groups.google.com for group membership checking:

  1. GroupProvider redirects client to google, requesting scope https://apps-apis.google.com/a/feeds/groups/
  1. Client allows access.
  1. Google redirects back to `/auth/groups/provider` k8s with an access token.
  1. GroupProvider uses token to read group membership from `https://apps-apis.google.com/a/feeds/group/2.0/{domain}/{group name}/member` once for each possible group (get from `AuthorizationProvider.GetRelevantGroups`).
  1. GroupProvider returns list of the groups that user belongs to.

## Notes

### Revokation

The lifetime of Tokens probably need to be long enough (>1d) to allow for infrequent re-auth.  But some organizations
need to be able to revoke a user or group membership promptly (<1h).  So, there may need to be an API action or special
file on the apiserver machines to revoke the Token of a user.

### Providers and API Plugins Security Comparison
This document refers to _API Plugins_, like discussed in [#1355](https://github.com/GoogleCloudPlatform/kubernetes/pull/1355).
API plugins are optional code which
  - can run in a different process from the apiserver
  - can register a new resource type (e.g. a Build Controller), 
  - handle APIserver requests for the resource paths (e.g. `/api/v1beta3/buildController`) for that resource.

Providers such as IdentityProvider, GroupProvider, and AuthorizationProvider might or might not also define new REST
resources (with AuthorizationProvider likely to do so.)  Since that code provides core acccess control functionality,
there is little reason to consider it less trusted than the APIserver.

Additionally, at least some amount of any Provider runs in APIserver itself, and could do unsafe accesses to APIserver memory.
While the provider could delegate most of its functionality to another process, there is little security benefit to
doing so.

In the case of non-Provider API Plugins, like a Build Controller, or a ShardController (the example given in #1355), the
contributed code could reasonably be treated as less trusted.  It typically makes API calls on other objects (e.g. pods) which are
subject to authorization.  In this case, there is some security benefit to making the contributed code run completely in
a separate process from the API server.

### Handling Auth for Resources added by API Plugins

Either the APIserver could always handle auth, or backends could handle auth for accesses to their resources.
  - If the APIserver always handles auth:
    - there needs to be a secure trusted channel between the APIserver and its backends, to prevent unauthorized actions from reaching the backend.
    - a trusted channel is needed in any case to ensure the right backend is contacted every time.
  - If the backends handle auth for their resources:
    - there will be considerable code duplication, unless a go library is written for the necessary authentication, authorization, caching, and selector evaluation.
    - there may be some memory duplication as the APIservers cache authorization decisions and indices needs to efficiently make authorization decisions.

The APIserver could delegate requests by proxying them, or by redirecting them to the appropriate backend.
  - If the APIserver proxies
    - it needs to hold a bit of state for each open request
    - it needs a trusted secure channel with its backends. This could be accomplished using SSL and self-signed certs for all plugins and apiserver, exchanged at config time.
  - If the APIserver redirects
    - if it wants to do auth, then it needs to attach a header or parameter to the redirected request proving that the action was authorized.  A special case of this is where the header/parameter is an Oauth token scoped to allow just that action.

This document assumes that the APIserver proxies requests to plugins after authenticating and authorizing them.  It should be possible to change that later.

### Future work

AuditLoggingProvider.    May need IP, time of request, and other attributed added to Attributes.

AdmissionControlProvider.  Takes the user, groups, namespace, and Resource size of the objects being created or deleted,
and decides if the request is within quota.

Service Accounts for Pods.  Depending on the environment, these might be accounts that are handled by the same
bedrock IdenityProvider as "user", or they might need to be specific to Kubernetes.  We may want to automate process or
generating Tokens for Pods and making those accessible to K8s clients running in containers.

DoS prevention.
