package ubiquity

import (
	"crypto/x509"
	"fmt"
	"time"

	"github.com/cloudflare/cfssl/helpers"
)

// DeprecationSeverity encodes the severity of a deprecation policy
type DeprecationSeverity int

const (
	// None indicates there is no deprecation
	None DeprecationSeverity = iota
	// Low indicates the deprecation policy won't affect user experience
	Low
	// Medium indicates the deprecation policy will affect user experience
	// either in a minor way or for a limited scope of users.
	Medium
	// High indicates the deprecation policy will strongly affect user experience
	High
)

// SHA1DeprecationPolicy encodes how a platform deprecates the support of SHA1
type SHA1DeprecationPolicy struct {
	// the name of platform
	Platform string `json:"platform"`
	// policy severity, policies of the same platform will only trigger the one of highest severity
	Severity DeprecationSeverity `json:"severity"`
	// a human readable message describing the deprecation effects
	Description string `json:"description"`
	// the date when the policy is effective. zero value means effective immediately
	EffectiveDate time.Time `json:"effective_date"`
	// the expiry deadline indicates the latest date which a end-entity
	// certificate with SHA1 can be valid through.
	ExpiryDeadline time.Time `json:"expiry_deadline"`
	// the date beyond which SHA1 cert should not be issued.
	NeverIssueAfter time.Time `json:"never_issue_after"`
}

// SHA1DeprecationPolicys ia a list of various SHA1DeprecationPolicy's
// proposed by major browser producers
var SHA1DeprecationPolicys = []SHA1DeprecationPolicy{
	// Chrome:
	//   if the leaf certificate expires between 01-01-2016 and 01-01-2017
	//   and the chain (excluding root) contains SHA-1 cert, show "minor errors".
	{
		Platform:       "Google Chrome",
		Description:    "shows the SSL connection has minor problems",
		Severity:       Medium,
		ExpiryDeadline: time.Date(2016, time.January, 1, 0, 0, 0, 0, time.UTC),
	},
	// Chrome:
	//   if the leaf certificate expires after Jan. 1st 2017
	//   and the chain (excluding root) contains SHA-1 cert, show "untrusted SSL".
	{
		Platform:       "Google Chrome",
		Description:    "shows the SSL connection is untrusted",
		Severity:       High,
		ExpiryDeadline: time.Date(2017, time.January, 1, 0, 0, 0, 0, time.UTC),
	},
	// Mozilla Firefox:
	//   if the leaf certificate expires after Jan. 1st 2017, and
	//   the chain (excluding root) contains SHA-1 cert, show a warning in the developer console.
	{
		Platform:       "Mozilla Firefox",
		Description:    "gives warning in the developer console",
		Severity:       Low,
		ExpiryDeadline: time.Date(2017, time.January, 1, 0, 0, 0, 0, time.UTC),
	},
	// Mozilla Firefox:
	//   if a new certificate is issued after Jan. 1st 2016, and
	//   it is a SHA-1 cert, reject it.
	{
		Platform:        "Mozilla Firefox",
		Description:     "shows the SSL connection is untrusted",
		Severity:        Medium,
		EffectiveDate:   time.Date(2016, time.January, 1, 0, 0, 0, 0, time.UTC),
		NeverIssueAfter: time.Date(2016, time.January, 1, 0, 0, 0, 0, time.UTC),
	},
	// Mozilla Firefox:
	//   deprecate all valid SHA-1 cert chain on Jan. 1st 2017
	{
		Platform:       "Mozilla Firefox",
		Description:    "shows the SSL connection is untrusted",
		Severity:       High,
		EffectiveDate:  time.Date(2017, time.January, 1, 0, 0, 0, 0, time.UTC),
		ExpiryDeadline: time.Date(2017, time.January, 1, 0, 0, 0, 0, time.UTC),
	},
	// Microsoft Windows:
	//   deprecate all valid SHA-1 cert chain on Jan. 1st 2017
	{
		Platform:       "Microsoft Windows Vista and later",
		Description:    "shows the SSL connection is untrusted",
		Severity:       High,
		EffectiveDate:  time.Date(2017, time.January, 1, 0, 0, 0, 0, time.UTC),
		ExpiryDeadline: time.Date(2017, time.January, 1, 0, 0, 0, 0, time.UTC),
	},
}

// Flag returns whether the policy flags the cert chain as deprecated for matching its deprecation criteria
func (p SHA1DeprecationPolicy) Flag(chain []*x509.Certificate) bool {
	leaf := chain[0]
	if time.Now().After(p.EffectiveDate) {

		// Reject newly issued leaf certificate with SHA-1 after the specified deadline.
		if !p.NeverIssueAfter.IsZero() && leaf.NotBefore.After(p.NeverIssueAfter) {
			// Check hash algorithm of non-root leaf cert.
			if len(chain) > 1 && helpers.HashAlgoString(leaf.SignatureAlgorithm) == "SHA1" {
				return true
			}
		}

		// Reject certificate chain with SHA-1 that are still valid after expiry deadline.
		if !p.ExpiryDeadline.IsZero() && leaf.NotAfter.After(p.ExpiryDeadline) {
			// Check hash algorithm of non-root certs.
			for i, cert := range chain {
				if i < len(chain)-1 {
					if helpers.HashAlgoString(cert.SignatureAlgorithm) == "SHA1" {
						return true
					}
				}
			}
		}
	}

	return false
}

// SHA1DeprecationMessages returns a list of human-readable messages. Each message describes
// how one platform rejects the chain based on SHA1 deprecation policies.
func SHA1DeprecationMessages(chain []*x509.Certificate) []string {
	// record the most severe deprecation policy by each platform
	selectedPolicies := map[string]SHA1DeprecationPolicy{}
	for _, policy := range SHA1DeprecationPolicys {
		if policy.Flag(chain) {
			// only keep the policy with highest severity
			if selectedPolicies[policy.Platform].Severity < policy.Severity {
				selectedPolicies[policy.Platform] = policy
			}
		}
	}
	// build the message list
	list := []string{}
	for _, policy := range selectedPolicies {
		if policy.Severity > None {
			list = append(list, fmt.Sprintf("%s %s due to SHA-1 deprecation", policy.Platform, policy.Description))
		}
	}
	return list
}
