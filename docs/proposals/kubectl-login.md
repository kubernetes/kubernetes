# Kubectl Login Subcommand

**Authors**: Eric Chiang (@ericchiang)

## Goals

`kubectl login` is an entrypoint for any user attempting to connect to an
existing server. It should provide a more tailored experience than the existing
`kubectl config` including config validation, auth challenges, and discovery.

Short term the subcommand should recognize and attempt to help:

* New users with an empty configuration trying to connect to a server.
* Users with no credentials, by prompt for any required information.
* Fully configured users who want to validate credentials.
* Users trying to switch servers.
* Users trying to reauthenticate as the same user because credentials have expired.
* Authenticate as a different user to the same server.

Long term `kubectl login` should enable authentication strategies to be
discoverable from a master to avoid the end-user having to know how their
sysadmin configured the Kubernetes cluster.

## Design

The "login" subcommand helps users move towards a fully functional kubeconfig by
evaluating the current state of the kubeconfig and trying to prompt the user for
and validate the necessary information to login to the kubernetes cluster.

This is inspired by a similar tools such as:

 * [os login](https://docs.openshift.org/latest/cli_reference/get_started_cli.html#basic-setup-and-login)
 * [gcloud auth login](https://cloud.google.com/sdk/gcloud/reference/auth/login)
 * [aws configure](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

The steps taken are:

1. If no cluster configured, prompt user for cluster information.
2. If no user is configured, discover the authentication strategies supported by the API server.
3. Prompt the user for some information based on the authentication strategy they choose.
4. Attempt to login as a user, including authentication challenges such as OAuth2 flows, and display user info.

Importantly, each step is skipped if the existing configuration is validated or
can be supplied without user interaction (refreshing an OAuth token, redeeming
a Kerberos ticket, etc.). Users with fully configured kubeconfigs will only see
the user they're logged in as, useful for opaque credentials such as X509 certs
or bearer tokens.

The command differs from `kubectl config` by:

* Communicating with the API server to determine if the user is supplying valid auth events.
* Validating input and being opinionated about the input it asks for.
* Triggering authentication challenges for example:
  * Basic auth: Actually try to communicate with the API server.
  * OpenID Connect: Create an OAuth2 redirect.

However `kubectl login` should still be seen as a supplement to, not a
replacement for, `kubectl config` by helping validate any kubeconfig generated
by the latter command.

## Credential validation

When clusters utilize authorization plugins access decisions are based on the
correct configuration of an auth-N plugin, an auth-Z plugin, and client side
credentials. Being rejected then begs several questions. Is the user's
kubeconfig misconfigured? Is the authorization plugin setup wrong? Is the user
authenticating as a different user than the one they assume?

To help `kubectl login` diagnose misconfigured credentials, responses from the
API server to authenticated requests SHOULD include the `Authentication-Info`
header as defined in [RFC 7615](https://tools.ietf.org/html/rfc7615). The value
will hold name value pairs for `username` and `uid`. Since usernames and IDs
can be arbitrary strings, these values will be escaped using the `quoted-string`
format noted in the RFC.

```
HTTP/1.1 200 OK
Authentication-Info: username="janedoe@example.com", uid="123456"
```

If the user successfully authenticates this header will be set, regardless of
auth-Z decisions. For example a 401 Unauthorized (user didn't provide valid
credentials) would lack this header, while a 403 Forbidden response would
contain it.

## Authentication discovery

A long term goal of `kubectl login` is to facilitate a customized experience
for clusters configured with different auth providers. This will require some
way for the API server to indicate to `kubectl` how a user is expected to
login.

Currently, this document doesn't propose a specific implementation for
discovery. While it'd be preferable to utilize an existing standard (such as the
`WWW-Authenticate` HTTP header), discovery may require a solution custom to the
API server, such as an additional discovery endpoint with a custom type.

## Use in non-interactive session

For the initial implementation, if `kubectl login` requires prompting and is
called from a non-interactive session (determined by if the session is using a
TTY) it errors out, recommending using `kubectl config` instead. In future
updates `kubectl login` may include options for non-interactive sessions so
auth strategies which require custom behavior not built into `kubectl config`,
such as the exchanges in Kerberos or OpenID Connect, can be triggered from
scripts.

## Examples

If kubeconfig isn't configured, `kubectl login` will attempt to fully configure
and validate the client's credentials.

```
$ kubectl login
Cluster URL []: https://172.17.4.99:443
Cluster CA [(defaults to host certs)]: ${PWD}/ssl/ca.pem
Cluster Name ["cluster-1"]:

The kubernetes server supports the following methods:

  1. Bearer token
  2. Username and password
  3. Keystone
  4. OpenID Connect
  5. TLS client certificate

Enter login method [1]: 4

Logging in using OpenID Connect.

Issuer ["valuefromdiscovery"]: https://accounts.google.com
Issuer CA [(defaults to host certs)]:
Scopes ["profile email"]:
Client ID []: client@localhost:foobar
Client Secret []: *****

Open the following address in a browser.

    https://accounts.google.com/o/oauth2/v2/auth?redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scopes=openid%20email&access_type=offline&...

Enter security code: ****

Logged in as "janedoe@gmail.com"
```

Human readable names are provided by a combination of the auth providers
understood by `kubectl login` and the authenticator discovery. For instance,
Keystone uses basic auth credentials in the same way as a static user file, but
if the discovery indicates that the Keystone plugin is being used it should be
presented to the user differently.

Users with configured credentials will simply auth against the API server and see
who they are. Running this command again simply validates the user's credentials.

```
$ kubectl login
Logged in as "janedoe@gmail.com"
```

Users who are halfway through the flow will start where they left off. For
instance if a user has configured the cluster field but on a user field, they will
be prompted for credentials.

```
$ kubectl login
No auth type configured. The kubernetes server supports the following methods:

  1. Bearer token
  2. Username and password
  3. Keystone
  4. OpenID Connect
  5. TLS client certificate

Enter login method [1]: 2

Logging in with basic auth. Enter the following fields.

Username: janedoe
Password: ****

Logged in as "janedoe@gmail.com"
```

Users who wish to switch servers can provide the `--switch-cluster` flag which
will prompt the user for new cluster details and switch the current context. It
behaves identically to `kubectl login` when a cluster is not set.

```
$ kubectl login --switch-cluster
# ...
```

Switching users goes through a similar flow attempting to prompt the user for
new credentials to the same server.

```
$ kubectl login --switch-user
# ...
```

## Work to do

Phase 1:

* Provide a simple dialog for configuring authentication.
* Kubectl can trigger authentication actions such as trigging OAuth2 redirects.
* Validation of user credentials thought the `Authentication-Info` endpoint.

Phase 2:

* Update proposal with auth provider discovery mechanism.
* Customize dialog using discovery data.

Further improvements will require adding more authentication providers, and
adapting existing plugins to take advantage of challenge based authentication.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubectl-login.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
