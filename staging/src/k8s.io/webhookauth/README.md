# webhookauth

Verification of KEP-6060 admission webhook bearer tokens.

An admission webhook receives a bearer token minted by the API server for the
webhook's ServiceAccount. This module verifies that token — signature, issuer,
audience, expiry, and the `admissionReviewAPIGroups` attestation — so a webhook
can trust that a request genuinely came from the API server and is authorized for
the API group under review.

The only public package is [`admissionhttp`](./admissionhttp). Everything else
lives under `internal/` and is not importable by consumers.

## Choosing an entry point

Pick the entry point that matches how your webhook receives requests. In both
cases the verification **mode** is selected by option *presence*, never by a
nil/zero value:

| Your webhook…                                             | Use                                   |
| -------------------------------------------------------- | ------------------------------------- |
| is a plain `net/http` server (you mount our handler)     | `admissionhttp.WithTokenVerification` |
| already decoded the review (controller-runtime, etc.)    | `admissionhttp.NewVerifier` + `Verify`|

| Verification mode        | Select it by…                          |
| ------------------------ | -------------------------------------- |
| in-cluster (zero-config) | passing **no** remote option (default) |
| remote (out-of-cluster)  | `admissionhttp.WithRemoteIssuer(...)`  |

No configuration does the right thing in-cluster; a partial remote config
(missing issuer or audience) is a hard construction error, never a silent
fallback.

The expected token **audience** is the webhook's own URL. A `net/http` handler
derives it from the first request; the already-decoded path has no request, so
supply it explicitly with `admissionhttp.WithAudience(...)` (issuer discovery
stays in-cluster). `WithAudience` cannot be combined with `WithRemoteIssuer`
(which already carries an audience).

| Consumer style           | in-cluster                                | remote                                  |
| ------------------------ | ----------------------------------------- | --------------------------------------- |
| `net/http` handler       | zero-config (audience from first request) | `WithRemoteIssuer`                      |
| already-decoded / CR     | `WithAudience`                            | `WithRemoteIssuer`                      |

> **Quirk — a wrong-but-derivable audience looks healthy.** Request derivation
> trusts the `<NAME>_SERVICE_PORT` env var and the request `Host`. If the derived
> audience (say, a mismatched port) differs from the audience the API server
> minted into the token, that derived value is still a *valid* string: it binds on
> the first request and `HealthCheck` reports **ready**, yet every token is denied
> fail-closed. The audience freezes on first bind, so there is no runtime
> recovery — redeploy with `WithAudience(...)` set to the token's true audience.
> `HealthCheck` can only confirm that *an* audience is bound, not that it is the
> *correct* one, so explicit `WithAudience` is the more robust choice.

## Usage

See the runnable examples on
[pkg.go.dev](https://pkg.go.dev/k8s.io/webhookauth/admissionhttp) —
`Example_rawHTTPWebhook` and `Example_controllerRuntimeStyle` — which stand up a
real signing issuer and verify real signatures end to end.

## Compatibility

Alpha, tracking [KEP-6060](https://kep.k8s.io/6060). The public API may change.

## Where does it come from?

`webhookauth` is synced from
https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/webhookauth.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later
synced here.

## Things you should NOT do

1. Directly modify any files in this repo. Those are copied from the main
   repository and synced here.
2. Expect compatibility. This repo is changing quickly in direct support of
   Kubernetes and the API isn't yet stable enough for API guarantees.
