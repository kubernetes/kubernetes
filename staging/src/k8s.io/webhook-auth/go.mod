// EXPLORATORY hand-written go.mod (KEP-6060 webhook-auth extraction).
// NOT canonical: the published/vendored go.mod MUST be produced by the
// sanctioned tooling (hack/update-vendor.sh + staging publishing rules).
// The indirect (// indirect) require block is intentionally omitted here and
// will be filled in by that tooling.

module k8s.io/webhook-auth

go 1.26.0

godebug default=go1.26

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
)
