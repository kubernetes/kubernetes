# OpenID Connect (OIDC) Authentication

## Usage

In your `.kube/config`, populate a `user` with an `auth-provider` named `oidc`. The `idp-ussuer-url` must be the same URL as the API Server's `oidc-issuer-url`. The client must be capable of creating ID tokens with an `aud` that is the API Server's `client-id`; this could mean that `kubectl` and the API Server have the same `client-id`, or it could use something like Google's [cross-client authentication](https://developers.google.com/identity/protocols/CrossClientAuth), which allows clients to mint tokens on behalf of another client (under certain restrictions).

If there's a `refresh-token` but no valid `id-token`, the plugin will attempt to obtain new short-lived credentials (i.e. the `id-token`) and if successful, will persist that in the config, and use them for the request. If the IdP also provides a new `refresh-token` that will be persisted as well.

Right now the only way to obtain a `refresh-token` or `id-token` is to run another application which can obtain them out-of-band, and copy them into your config. In the near future it is expected that there will be support for obtaining credentials by initiaiting authentication from `kubectl`.

## Configuration

| key | description | |
| --- | --- | --- |
| client-id | The client id of the kubectl client. | REQUIRED |
| client-secret | The client secret of the kubectl client. | REQUIRED |
| idp-issuer-url | The URL of the identity provider (IdP), i.e. the authentication server, eg. "https://accounts.google.com" | REQUIRED |
| idp-certificate-authority | Path to a cert file for the certificate authority used to communication with the IdP. If neither this field nor `idp-certificate-authority-data` are present, the hosts' trusted CAs will be used. | |
| idp-certificate-authority-data | PEM-encoded certificate authority certs. Overrides `idp-certificate-authority` | |
| extra-scopes | Additional scopes to be added to the authentication request's default scopes. Default scopes are `openid`, `email` and `profile`. | |
| id-token | This is the short-lived credentials, used to authenticate requests to the API Server. This contents of this field must be protected and shared with no one besides the end user. | |
| refresh-token | The OIDC/OAuth2 refresh token. These are long-lived credentials, so extra care must be taken to protect the contents of this field. | |
