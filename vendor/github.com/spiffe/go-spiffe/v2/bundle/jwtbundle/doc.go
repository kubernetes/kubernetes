// Package jwtbundle provides JWT bundle related functionality.
//
// A bundle represents a collection of JWT authorities, i.e., those that
// are used to authenticate SPIFFE JWT-SVIDs.
//
// You can create a new bundle for a specific trust domain:
//
//	td := spiffeid.RequireTrustDomainFromString("example.org")
//	bundle := jwtbundle.New(td)
//
// Or you can load it from disk:
//
//	td := spiffeid.RequireTrustDomainFromString("example.org")
//	bundle := jwtbundle.Load(td, "bundle.jwks")
//
// The bundle can be initialized with JWT authorities:
//
//	td := spiffeid.RequireTrustDomainFromString("example.org")
//	var jwtAuthorities map[string]crypto.PublicKey = ...
//	bundle := jwtbundle.FromJWTAuthorities(td, jwtAuthorities)
//
// In addition, you can add JWT authorities to the bundle:
//
//	var keyID string = ...
//	var publicKey crypto.PublicKey = ...
//	bundle.AddJWTAuthority(keyID, publicKey)
//
// Bundles can be organized into a set, keyed by trust domain:
//
//	set := jwtbundle.NewSet()
//	set.Add(bundle)
//
// A Source is source of JWT bundles for a trust domain. Both the Bundle
// and Set types implement Source:
//
//	// Initialize the source from a bundle or set
//	var source jwtbundle.Source = bundle
//	// ... or ...
//	var source jwtbundle.Source = set
//
//	// Use the source to query for bundles by trust domain
//	bundle, err := source.GetJWTBundleForTrustDomain(td)
package jwtbundle
