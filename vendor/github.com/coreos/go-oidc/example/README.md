# Examples

These are example uses of the oidc package. Each requires a Google account and the client ID and secret of a registered OAuth2 application. To create one:

1. Visit your [Google Developer Console][google-developer-console].
2. Click "Credentials" on the left column.
3. Click the "Create credentials" button followed by "OAuth client ID".
4. Select "Web application" and add "http://127.0.0.1:5556/auth/google/callback" as an authorized redirect URI.
5. Click create and add the printed client ID and secret to your environment using the following variables:

```
GOOGLE_OAUTH2_CLIENT_ID
GOOGLE_OAUTH2_CLIENT_SECRET
```

Finally run the examples using the Go tool and navigate to http://127.0.0.1:5556.

```
go run ./example/idtoken/app.go
```
[google-developer-console]: https://console.developers.google.com/apis/dashboard
