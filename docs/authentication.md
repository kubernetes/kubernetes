# Authentication Plugins

Kubernetes uses tokens to authenticate users for API calls.

Authentication is enabled by passing the `--token_auth_file=SOMEFILE` option
to apiserver.  Currently, tokens last indefinitely, and the token list cannot
be changed without restarting apiserver.  We plan in the future for tokens to
be short-lived, and to be generated as needed rather than stored in a file.

The token file format is implemented in `plugin/pkg/auth/authenticator/token/tokenfile/...`
and is a csv file with 3 columns: token, user name, user uid.

## Plugin Development

We plan for the Kubernetes API server to issue tokens
after the user has been (re)authenticated by a *bedrock* authentication
provider external to Kubernetes.  We plan to make it easy to develop modules
that interface between kubernetes and a bedrock authentication provider (e.g.
github.com, google.com, enterprise directory, kerberos, etc.)
