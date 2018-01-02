// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package system

import (
	"crypto/x509"
	"errors"
	"syscall"
	"unsafe"
)

// Creates a new *syscall.CertContext representing the leaf certificate in an in-memory
// certificate store containing itself and all of the intermediate certificates specified
// in the opts.Intermediates CertPool.
//
// A pointer to the in-memory store is available in the returned CertContext's Store field.
// The store is automatically freed when the CertContext is freed using
// syscall.CertFreeCertificateContext.
func createStoreContext(leaf *Certificate, opts *VerifyOptions) (*syscall.CertContext, error) {
	var storeCtx *syscall.CertContext

	leafCtx, err := syscall.CertCreateCertificateContext(syscall.X509_ASN_ENCODING|syscall.PKCS_7_ASN_ENCODING, &leaf.Raw[0], uint32(len(leaf.Raw)))
	if err != nil {
		return nil, err
	}
	defer syscall.CertFreeCertificateContext(leafCtx)

	handle, err := syscall.CertOpenStore(syscall.CERT_STORE_PROV_MEMORY, 0, 0, syscall.CERT_STORE_DEFER_CLOSE_UNTIL_LAST_FREE_FLAG, 0)
	if err != nil {
		return nil, err
	}
	defer syscall.CertCloseStore(handle, 0)

	err = syscall.CertAddCertificateContextToStore(handle, leafCtx, syscall.CERT_STORE_ADD_ALWAYS, &storeCtx)
	if err != nil {
		return nil, err
	}

	if opts.Intermediates != nil {
		for _, intermediate := range opts.Intermediates.certs {
			ctx, err := syscall.CertCreateCertificateContext(syscall.X509_ASN_ENCODING|syscall.PKCS_7_ASN_ENCODING, &intermediate.Raw[0], uint32(len(intermediate.Raw)))
			if err != nil {
				return nil, err
			}

			err = syscall.CertAddCertificateContextToStore(handle, ctx, syscall.CERT_STORE_ADD_ALWAYS, nil)
			syscall.CertFreeCertificateContext(ctx)
			if err != nil {
				return nil, err
			}
		}
	}

	return storeCtx, nil
}

// extractSimpleChain extracts the final certificate chain from a CertSimpleChain.
func extractSimpleChain(simpleChain **syscall.CertSimpleChain, count int) (chain []*Certificate, err error) {
	if simpleChain == nil || count == 0 {
		return nil, errors.New("x509: invalid simple chain")
	}

	simpleChains := (*[1 << 20]*syscall.CertSimpleChain)(unsafe.Pointer(simpleChain))[:]
	lastChain := simpleChains[count-1]
	elements := (*[1 << 20]*syscall.CertChainElement)(unsafe.Pointer(lastChain.Elements))[:]
	for i := 0; i < int(lastChain.NumElements); i++ {
		// Copy the buf, since ParseCertificate does not create its own copy.
		cert := elements[i].CertContext
		encodedCert := (*[1 << 20]byte)(unsafe.Pointer(cert.EncodedCert))[:]
		buf := make([]byte, cert.Length)
		copy(buf, encodedCert[:])
		parsedCert, err := ParseCertificate(buf)
		if err != nil {
			return nil, err
		}
		chain = append(chain, parsedCert)
	}

	return chain, nil
}

// checkChainTrustStatus checks the trust status of the certificate chain, translating
// any errors it finds into Go errors in the process.
func checkChainTrustStatus(c *Certificate, chainCtx *syscall.CertChainContext) error {
	if chainCtx.TrustStatus.ErrorStatus != syscall.CERT_TRUST_NO_ERROR {
		status := chainCtx.TrustStatus.ErrorStatus
		switch status {
		case syscall.CERT_TRUST_IS_NOT_TIME_VALID:
			return CertificateInvalidError{c, Expired}
		default:
			return UnknownAuthorityError{c, nil, nil}
		}
	}
	return nil
}

// checkChainSSLServerPolicy checks that the certificate chain in chainCtx is valid for
// use as a certificate chain for a SSL/TLS server.
func checkChainSSLServerPolicy(c *Certificate, chainCtx *syscall.CertChainContext, opts *VerifyOptions) error {
	servernamep, err := syscall.UTF16PtrFromString(opts.DNSName)
	if err != nil {
		return err
	}
	sslPara := &syscall.SSLExtraCertChainPolicyPara{
		AuthType:   syscall.AUTHTYPE_SERVER,
		ServerName: servernamep,
	}
	sslPara.Size = uint32(unsafe.Sizeof(*sslPara))

	para := &syscall.CertChainPolicyPara{
		ExtraPolicyPara: uintptr(unsafe.Pointer(sslPara)),
	}
	para.Size = uint32(unsafe.Sizeof(*para))

	status := syscall.CertChainPolicyStatus{}
	err = syscall.CertVerifyCertificateChainPolicy(syscall.CERT_CHAIN_POLICY_SSL, chainCtx, para, &status)
	if err != nil {
		return err
	}

	// TODO(mkrautz): use the lChainIndex and lElementIndex fields
	// of the CertChainPolicyStatus to provide proper context, instead
	// using c.
	if status.Error != 0 {
		switch status.Error {
		case syscall.CERT_E_EXPIRED:
			return CertificateInvalidError{c, Expired}
		case syscall.CERT_E_CN_NO_MATCH:
			return HostnameError{c, opts.DNSName}
		case syscall.CERT_E_UNTRUSTEDROOT:
			return UnknownAuthorityError{c, nil, nil}
		default:
			return UnknownAuthorityError{c, nil, nil}
		}
	}

	return nil
}

// Note(kyle): not sure how this works on windows, or if this does.
func initSystemRoots() []*x509.Certificate {
	return nil
}
