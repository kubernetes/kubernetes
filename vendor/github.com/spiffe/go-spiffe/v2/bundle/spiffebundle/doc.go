// Package spiffebundle provides SPIFFE bundle related functionality.
//
// A bundle represents a SPIFFE bundle, a collection authorities for
// authenticating SVIDs.
//
// You can create a new bundle for a specific trust domain:
//
//	td := spiffeid.RequireTrustDomainFromString("example.org")
//	bundle := spiffebundle.New(td)
//
// Or you can load it from disk:
//
//	td := spiffeid.RequireTrustDomainFromString("example.org")
//	bundle := spiffebundle.Load(td, "bundle.json")
//
// The bundle can be initialized with X.509 or JWT authorities:
//
//	td := spiffeid.RequireTrustDomainFromString("example.org")
//
//	var x509Authorities []*x509.Certificate = ...
//	bundle := spiffebundle.FromX509Authorities(td, x509Authorities)
//	// ... or ...
//	var jwtAuthorities map[string]crypto.PublicKey = ...
//	bundle := spiffebundle.FromJWTAuthorities(td, jwtAuthorities)
//
// In addition, you can add authorities to the bundle:
//
//	var x509CA *x509.Certificate = ...
//	bundle.AddX509Authority(x509CA)
//	var keyID string = ...
//	var publicKey crypto.PublicKey = ...
//	bundle.AddJWTAuthority(keyID, publicKey)
//
// Bundles can be organized into a set, keyed by trust domain:
//
//	set := spiffebundle.NewSet()
//	set.Add(bundle)
//
// A Source is source of bundles for a trust domain. Both the
// Bundle and Set types implement Source:
//
//	// Initialize the source from a bundle or set
//	var source spiffebundle.Source = bundle
//	// ... or ...
//	var source spiffebundle.Source = set
//
//	// Use the source to query for X.509 bundles by trust domain
//	bundle, err := source.GetBundleForTrustDomain(td)
//
// Additionally the Bundle and Set types also implement the x509bundle.Source and jwtbundle.Source interfaces:
//
//	// As an x509bundle.Source...
//	var source x509bundle.Source = bundle // or set
//	x509Bundle, err := source.GetX509BundleForTrustDomain(td)
//
//	// As a jwtbundle.Source...
//	var source jwtbundle.Source = bundle // or set
//	jwtBundle, err := source.GetJWTBundleForTrustDomain(td)
package spiffebundle
