/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package authority

import (
	"crypto/x509"
	"fmt"
	"sort"
	"time"

	capi "k8s.io/api/certificates/v1beta1"
)

// SigningPolicy validates a CertificateRequest before it's signed by the
// CertificateAuthority. It may default or otherwise mutate a certificate
// template.
type SigningPolicy interface {
	// not-exporting apply forces signing policy implementations to be internal
	// to this package.
	apply(template *x509.Certificate) error
}

// PermissiveSigningPolicy is the signing policy historically used by the local
// signer.
//
//  * It forwards all SANs from the original signing request.
//  * It sets allowed usages as configured in the policy.
//  * It sets NotAfter based on the TTL configured in the policy.
//  * It zeros all extensions.
//  * It sets BasicConstraints to true.
//  * It sets IsCA to false.
type PermissiveSigningPolicy struct {
	// TTL is the certificate TTL. It's used to calculate the NotAfter value of
	// the certificate.
	TTL time.Duration
	// Usages are the allowed usages of a certificate.
	Usages []capi.KeyUsage
}

func (p PermissiveSigningPolicy) apply(tmpl *x509.Certificate) error {
	usage, extUsages, err := keyUsagesFromStrings(p.Usages)
	if err != nil {
		return err
	}
	tmpl.KeyUsage = usage
	tmpl.ExtKeyUsage = extUsages
	tmpl.NotAfter = tmpl.NotBefore.Add(p.TTL)

	tmpl.ExtraExtensions = nil
	tmpl.Extensions = nil
	tmpl.BasicConstraintsValid = true
	tmpl.IsCA = false

	return nil
}

var keyUsageDict = map[capi.KeyUsage]x509.KeyUsage{
	capi.UsageSigning:           x509.KeyUsageDigitalSignature,
	capi.UsageDigitalSignature:  x509.KeyUsageDigitalSignature,
	capi.UsageContentCommitment: x509.KeyUsageContentCommitment,
	capi.UsageKeyEncipherment:   x509.KeyUsageKeyEncipherment,
	capi.UsageKeyAgreement:      x509.KeyUsageKeyAgreement,
	capi.UsageDataEncipherment:  x509.KeyUsageDataEncipherment,
	capi.UsageCertSign:          x509.KeyUsageCertSign,
	capi.UsageCRLSign:           x509.KeyUsageCRLSign,
	capi.UsageEncipherOnly:      x509.KeyUsageEncipherOnly,
	capi.UsageDecipherOnly:      x509.KeyUsageDecipherOnly,
}

var extKeyUsageDict = map[capi.KeyUsage]x509.ExtKeyUsage{
	capi.UsageAny:             x509.ExtKeyUsageAny,
	capi.UsageServerAuth:      x509.ExtKeyUsageServerAuth,
	capi.UsageClientAuth:      x509.ExtKeyUsageClientAuth,
	capi.UsageCodeSigning:     x509.ExtKeyUsageCodeSigning,
	capi.UsageEmailProtection: x509.ExtKeyUsageEmailProtection,
	capi.UsageSMIME:           x509.ExtKeyUsageEmailProtection,
	capi.UsageIPsecEndSystem:  x509.ExtKeyUsageIPSECEndSystem,
	capi.UsageIPsecTunnel:     x509.ExtKeyUsageIPSECTunnel,
	capi.UsageIPsecUser:       x509.ExtKeyUsageIPSECUser,
	capi.UsageTimestamping:    x509.ExtKeyUsageTimeStamping,
	capi.UsageOCSPSigning:     x509.ExtKeyUsageOCSPSigning,
	capi.UsageMicrosoftSGC:    x509.ExtKeyUsageMicrosoftServerGatedCrypto,
	capi.UsageNetscapeSGC:     x509.ExtKeyUsageNetscapeServerGatedCrypto,
}

// keyUsagesFromStrings will translate a slice of usage strings from the
// certificates API ("pkg/apis/certificates".KeyUsage) to x509.KeyUsage and
// x509.ExtKeyUsage types.
func keyUsagesFromStrings(usages []capi.KeyUsage) (x509.KeyUsage, []x509.ExtKeyUsage, error) {
	var keyUsage x509.KeyUsage
	var unrecognized []capi.KeyUsage
	extKeyUsages := make(map[x509.ExtKeyUsage]struct{})
	for _, usage := range usages {
		if val, ok := keyUsageDict[usage]; ok {
			keyUsage |= val
		} else if val, ok := extKeyUsageDict[usage]; ok {
			extKeyUsages[val] = struct{}{}
		} else {
			unrecognized = append(unrecognized, usage)
		}
	}

	var sorted sortedExtKeyUsage
	for eku := range extKeyUsages {
		sorted = append(sorted, eku)
	}
	sort.Sort(sorted)

	if len(unrecognized) > 0 {
		return 0, nil, fmt.Errorf("unrecognized usage values: %q", unrecognized)
	}

	return keyUsage, []x509.ExtKeyUsage(sorted), nil
}

type sortedExtKeyUsage []x509.ExtKeyUsage

func (s sortedExtKeyUsage) Len() int {
	return len(s)
}

func (s sortedExtKeyUsage) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s sortedExtKeyUsage) Less(i, j int) bool {
	return s[i] < s[j]
}
