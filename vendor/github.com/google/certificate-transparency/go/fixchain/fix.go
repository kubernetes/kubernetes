package fixchain

import (
	"encoding/pem"
	"net/http"

	"github.com/google/certificate-transparency/go/x509"
)

// Fix attempts to fix the certificate chain for the certificate that is passed
// to it, with respect to the given roots.  Fix returns a list of successfully
// constructed chains, and a list of errors it encountered along the way.  The
// presence of FixErrors does not mean the fix was unsuccessful.  Callers should
// check for returned chains to determine success.
func Fix(cert *x509.Certificate, chain []*x509.Certificate, roots *x509.CertPool, client *http.Client) ([][]*x509.Certificate, []*FixError) {
	fix := &toFix{
		cert:  cert,
		chain: newDedupedChain(chain),
		roots: roots,
		cache: newURLCache(client, false),
	}
	return fix.handleChain()
}

const maxChainLength = 20

type toFix struct {
	cert  *x509.Certificate
	chain *dedupedChain
	roots *x509.CertPool
	opts  *x509.VerifyOptions
	cache *urlCache
}

func (fix *toFix) handleChain() ([][]*x509.Certificate, []*FixError) {
	intermediates := x509.NewCertPool()
	for _, c := range fix.chain.certs {
		intermediates.AddCert(c)
	}

	fix.opts = &x509.VerifyOptions{
		Intermediates:     intermediates,
		Roots:             fix.roots,
		DisableTimeChecks: true,
		KeyUsages:         []x509.ExtKeyUsage{x509.ExtKeyUsageAny},
	}

	var retferrs []*FixError
	chains, ferrs := fix.constructChain()
	if ferrs != nil {
		retferrs = append(retferrs, ferrs...)
		chains, ferrs = fix.fixChain()
		if ferrs != nil {
			retferrs = append(retferrs, ferrs...)
		}
	}
	return chains, retferrs
}

func (fix *toFix) constructChain() ([][]*x509.Certificate, []*FixError) {
	chains, err := fix.cert.Verify(*fix.opts)
	if err != nil {
		return chains, []*FixError{
			&FixError{
				Type:  VerifyFailed,
				Cert:  fix.cert,
				Chain: fix.chain.certs,
				Error: err,
			},
		}
	}
	return chains, nil
}

// toFix.fixChain() tries to fix the certificate chain in the toFix struct for
// the cert in the toFix struct wrt the roots in the toFix struct.
// toFix.fixChain() uses the opts provided in the toFix struct to verify the
// chain, and uses the cache in the toFix struct to go and get any potentially
// missing intermediate certs.
// toFix.fixChain() returns a slice of valid and verified chains for this cert
// to the roots in the toFix struct, and a slice of the errors encountered
// during the fixing process.
func (fix *toFix) fixChain() ([][]*x509.Certificate, []*FixError) {
	var retferrs []*FixError

	// Ensure the leaf certificate is included as part of the certificate chain.
	dchain := *fix.chain
	dchain.addCertToFront(fix.cert)

	explored := make([]bool, len(dchain.certs))
	lookup := make(map[[hashSize]byte]int)
	for i, cert := range dchain.certs {
		lookup[hash(cert)] = i
	}

	// For each certificate in the given certificate chain...
	for i, cert := range dchain.certs {
		// If the chains from this certificate have already been built and
		// added to the pool of intermediates, skip.
		if explored[i] {
			continue
		}

		seen := make(map[[hashSize]byte]bool)
		// Build all the chains possible that begin from this certificate,
		// and add each certificate found along the way to the pool of
		// intermediates against which to verify fix.cert.  If the addition of
		// these intermediates causes chains for fix.cert to be verified,
		// fix.augmentIntermediates() will return those chains.
		chains, ferrs := fix.augmentIntermediates(cert, 1, seen)
		if ferrs != nil {
			retferrs = append(retferrs, ferrs...)
		}
		// If adding certs from the chains steming from this cert resulted in
		// successful verification of chains for fix.cert to fix.root, return
		// the chains.
		if chains != nil {
			return chains, retferrs
		}

		// Mark any seen certs that match certs in the original chain as already
		// explored.
		for certHash := range seen {
			index, ok := lookup[certHash]
			if ok {
				explored[index] = true
			}
		}
	}

	return nil, append(retferrs, &FixError{
		Type:  FixFailed,
		Cert:  fix.cert,
		Chain: fix.chain.certs,
	})
}

// TODO(katjoyce): Extend fixing algorithm to build all of the chains for
// toFix.cert and log all of the resulting intermediates.

// toFix.augmentIntermediates() builds all possible chains that stem from the
// given cert, and adds every certificate it finds in these chains to the pool
// of intermediate certs in toFix.opts.  Every time a new certificate is added
// to this pool, it tries to re-verify toFix.cert wrt toFix.roots.
// If this verification is ever successful, toFix.augmentIntermediates() returns
// the verified chains for toFix.cert wrt toFix.roots.  Also returned are any
// errors that were encountered along the way.
//
// toFix.augmentIntermediates() builds all possible chains from cert by using a
// recursive algorithm on the urls in the AIA information of each certificate
// discovered. length represents the position of the current given cert in the
// larger chain, and is used to impose a max length to which chains can be
// explored.  seen is a slice in which all certs that are encountered during the
// search are noted down.
func (fix *toFix) augmentIntermediates(cert *x509.Certificate, length int, seen map[[hashSize]byte]bool) ([][]*x509.Certificate, []*FixError) {
	// If this cert takes the chain past maxChainLength, or if this cert has
	// already been explored, return.
	if length > maxChainLength || seen[hash(cert)] {
		return nil, nil
	}
	// Mark this cert as already explored.
	seen[hash(cert)] = true

	// Add this cert to the pool of intermediates.  If this results in successful
	// verification of one or more chains for fix.cert, return the chains.
	fix.opts.Intermediates.AddCert(cert)
	chains, err := fix.cert.Verify(*fix.opts)
	if err == nil {
		return chains, nil
	}

	// For each url in the AIA information of cert, get the corresponding
	// certificates and recursively build the chains from those certificates,
	// adding every cert to the pool of intermdiates, running the verifier at
	// every cert addition, and returning verified chains of fix.cert as soon
	// as thay are found.
	var retferrs []*FixError
	for _, url := range cert.IssuingCertificateURL {
		icerts, ferr := fix.getIntermediates(url)
		if ferr != nil {
			retferrs = append(retferrs, ferr)
		}

		for _, icert := range icerts {
			chains, ferrs := fix.augmentIntermediates(icert, length+1, seen)
			if ferrs != nil {
				retferrs = append(retferrs, ferrs...)
			}
			if chains != nil {
				return chains, retferrs
			}
		}
	}
	return nil, retferrs
}

// Get the certs that correspond to the given url.
func (fix *toFix) getIntermediates(url string) ([]*x509.Certificate, *FixError) {
	var icerts []*x509.Certificate
	// PKCS#7 additions as (at time of writing) there is no standard Go PKCS#7
	// implementation
	r := urlReplacement(url)
	if r != nil {
		return r, nil
	}

	body, err := fix.cache.getURL(url)
	if err != nil {
		return nil, &FixError{
			Type:  CannotFetchURL,
			Cert:  fix.cert,
			Chain: fix.chain.certs,
			URL:   url,
			Error: err,
		}
	}

	icert, err := x509.ParseCertificate(body)
	if err != nil {
		s, _ := pem.Decode(body)
		if s != nil {
			icert, err = x509.ParseCertificate(s.Bytes)
		}
	}

	if err != nil {
		return nil, &FixError{
			Type:  ParseFailure,
			Cert:  fix.cert,
			Chain: fix.chain.certs,
			URL:   url,
			Bad:   body,
			Error: err,
		}
	}

	icerts = append(icerts, icert)
	return icerts, nil
}
