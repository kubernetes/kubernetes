// Package bundler implements certificate bundling functionality for
// CFSSL.
package bundler

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	goerr "errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/ubiquity"
)

// IntermediateStash contains the path to the directory where
// downloaded intermediates should be saved.
// When unspecified, downloaded intermediates are not saved.
var IntermediateStash string

// BundleFlavor is named optimization strategy on certificate chain selection when bundling.
type BundleFlavor string

const (
	// Optimal means the shortest chain with newest intermediates and
	// the most advanced crypto.
	Optimal BundleFlavor = "optimal"

	// Ubiquitous is aimed to provide the chain which is accepted
	// by the most platforms.
	Ubiquitous BundleFlavor = "ubiquitous"

	// Force means the bundler only verfiies the input as a valid bundle, not optimization is done.
	Force BundleFlavor = "force"
)

const (
	sha2Warning          = "The bundle contains certificates signed with advanced hash functions such as SHA2, which are problematic for certain operating systems, e.g. Windows XP SP2."
	ecdsaWarning         = "The bundle contains ECDSA signatures, which are problematic for certain operating systems, e.g. Windows XP, Android 2.2 and Android 2.3."
	expiringWarningStub  = "The bundle is expiring within 30 days."
	untrustedWarningStub = "The bundle may not be trusted by the following platform(s):"
	ubiquityWarning      = "Unable to measure bundle ubiquity: No platform metadata present."
)

// A Bundler contains the certificate pools for producing certificate
// bundles. It contains any intermediates and root certificates that
// should be used.
type Bundler struct {
	RootPool         *x509.CertPool
	IntermediatePool *x509.CertPool
	KnownIssuers     map[string]bool
	opts             options
}

type options struct {
	keyUsages []x509.ExtKeyUsage
}

var defaultOptions = options{
	keyUsages: []x509.ExtKeyUsage{
		x509.ExtKeyUsageAny,
	},
}

// An Option sets options such as allowed key usages, etc.
type Option func(*options)

// WithKeyUsages lets you set which Extended Key Usage values are acceptable. By
// default x509.ExtKeyUsageAny will be used.
func WithKeyUsages(usages ...x509.ExtKeyUsage) Option {
	return func(o *options) {
		o.keyUsages = usages
	}
}

// NewBundler creates a new Bundler from the files passed in; these
// files should contain a list of valid root certificates and a list
// of valid intermediate certificates, respectively.
func NewBundler(caBundleFile, intBundleFile string, opt ...Option) (*Bundler, error) {
	var caBundle, intBundle []byte
	var err error

	if caBundleFile != "" {
		log.Debug("Loading CA bundle: ", caBundleFile)
		caBundle, err = ioutil.ReadFile(caBundleFile)
		if err != nil {
			log.Errorf("root bundle failed to load: %v", err)
			return nil, errors.Wrap(errors.RootError, errors.ReadFailed, err)
		}
	}

	if intBundleFile != "" {
		log.Debug("Loading Intermediate bundle: ", intBundleFile)
		intBundle, err = ioutil.ReadFile(intBundleFile)
		if err != nil {
			log.Errorf("intermediate bundle failed to load: %v", err)
			return nil, errors.Wrap(errors.IntermediatesError, errors.ReadFailed, err)
		}
	}

	if IntermediateStash != "" {
		if _, err = os.Stat(IntermediateStash); err != nil && os.IsNotExist(err) {
			log.Infof("intermediate stash directory %s doesn't exist, creating", IntermediateStash)
			err = os.MkdirAll(IntermediateStash, 0755)
			if err != nil {
				log.Errorf("failed to create intermediate stash directory %s: %v",
					IntermediateStash, err)
				return nil, err
			}
			log.Infof("intermediate stash directory %s created", IntermediateStash)
		}
	}

	return NewBundlerFromPEM(caBundle, intBundle, opt...)

}

// NewBundlerFromPEM creates a new Bundler from PEM-encoded root certificates and
// intermediate certificates.
// If caBundlePEM is nil, the resulting Bundler can only do "Force" bundle.
func NewBundlerFromPEM(caBundlePEM, intBundlePEM []byte, opt ...Option) (*Bundler, error) {
	opts := defaultOptions
	for _, o := range opt {
		o(&opts)
	}

	log.Debug("parsing root certificates from PEM")
	roots, err := helpers.ParseCertificatesPEM(caBundlePEM)
	if err != nil {
		log.Errorf("failed to parse root bundle: %v", err)
		return nil, errors.New(errors.RootError, errors.ParseFailed)
	}

	log.Debug("parse intermediate certificates from PEM")
	intermediates, err := helpers.ParseCertificatesPEM(intBundlePEM)
	if err != nil {
		log.Errorf("failed to parse intermediate bundle: %v", err)
		return nil, errors.New(errors.IntermediatesError, errors.ParseFailed)
	}

	b := &Bundler{
		KnownIssuers:     map[string]bool{},
		IntermediatePool: x509.NewCertPool(),
		opts:             opts,
	}

	log.Debug("building certificate pools")

	// RootPool will be nil if caBundlePEM is nil, also
	// that translates to caBundleFile is "".
	// Systems root store will be used.
	if caBundlePEM != nil {
		b.RootPool = x509.NewCertPool()
	}

	for _, c := range roots {
		b.RootPool.AddCert(c)
		b.KnownIssuers[string(c.Signature)] = true
	}

	for _, c := range intermediates {
		b.IntermediatePool.AddCert(c)
		b.KnownIssuers[string(c.Signature)] = true
	}

	log.Debug("bundler set up")
	return b, nil
}

// VerifyOptions generates an x509 VerifyOptions structure that can be
// used for verifying certificates.
func (b *Bundler) VerifyOptions() x509.VerifyOptions {
	return x509.VerifyOptions{
		Roots:         b.RootPool,
		Intermediates: b.IntermediatePool,
		KeyUsages:     b.opts.keyUsages,
	}
}

// BundleFromFile takes a set of files containing the PEM-encoded leaf certificate
// (optionally along with some intermediate certs), the PEM-encoded private key
// and returns the bundle built from that key and the certificate(s).
func (b *Bundler) BundleFromFile(bundleFile, keyFile string, flavor BundleFlavor, password string) (*Bundle, error) {
	log.Debug("Loading Certificate: ", bundleFile)
	certsRaw, err := ioutil.ReadFile(bundleFile)
	if err != nil {
		return nil, errors.Wrap(errors.CertificateError, errors.ReadFailed, err)
	}

	var keyPEM []byte
	// Load private key PEM only if a file is given
	if keyFile != "" {
		log.Debug("Loading private key: ", keyFile)
		keyPEM, err = ioutil.ReadFile(keyFile)
		if err != nil {
			log.Debugf("failed to read private key: ", err)
			return nil, errors.Wrap(errors.PrivateKeyError, errors.ReadFailed, err)
		}
		if len(keyPEM) == 0 {
			log.Debug("key is empty")
			return nil, errors.Wrap(errors.PrivateKeyError, errors.DecodeFailed, err)
		}
	}

	return b.BundleFromPEMorDER(certsRaw, keyPEM, flavor, password)
}

// BundleFromPEMorDER builds a certificate bundle from the set of byte
// slices containing the PEM or DER-encoded certificate(s), private key.
func (b *Bundler) BundleFromPEMorDER(certsRaw, keyPEM []byte, flavor BundleFlavor, password string) (*Bundle, error) {
	log.Debug("bundling from PEM files")
	var key crypto.Signer
	var err error
	if len(keyPEM) != 0 {
		key, err = helpers.ParsePrivateKeyPEM(keyPEM)
		if err != nil {
			log.Debugf("failed to parse private key: %v", err)
			return nil, err
		}
	}

	certs, err := helpers.ParseCertificatesPEM(certsRaw)
	if err != nil {
		// If PEM doesn't work try DER
		var keyDER crypto.Signer
		var errDER error
		certs, keyDER, errDER = helpers.ParseCertificatesDER(certsRaw, password)
		// Only use DER key if no key read from file
		if key == nil && keyDER != nil {
			key = keyDER
		}
		if errDER != nil {
			log.Debugf("failed to parse certificates: %v", err)
			// If neither parser works pass along PEM error
			return nil, err
		}

	}
	if len(certs) == 0 {
		log.Debugf("no certificates found")
		return nil, errors.New(errors.CertificateError, errors.DecodeFailed)
	}

	log.Debugf("bundle ready")
	return b.Bundle(certs, key, flavor)
}

// BundleFromRemote fetches the certificate served by the server at
// serverName (or ip, if the ip argument is not the empty string). It
// is expected that the method will be able to make a connection at
// port 443. The certificate used by the server in this connection is
// used to build the bundle, which will necessarily be keyless.
func (b *Bundler) BundleFromRemote(serverName, ip string, flavor BundleFlavor) (*Bundle, error) {
	config := &tls.Config{
		RootCAs:    b.RootPool,
		ServerName: serverName,
	}

	// Dial by IP if present
	var dialName string
	if ip != "" {
		dialName = ip + ":443"
	} else {
		dialName = serverName + ":443"
	}

	log.Debugf("bundling from remote %s", dialName)

	dialer := &net.Dialer{Timeout: time.Duration(5) * time.Second}
	conn, err := tls.DialWithDialer(dialer, "tcp", dialName, config)
	var dialError string
	// If there's an error in tls.Dial, try again with
	// InsecureSkipVerify to fetch the remote bundle to (re-)bundle
	// with. If the bundle is indeed not usable (expired, mismatched
	// hostnames, etc.), report the error.  Otherwise, create a
	// working bundle and insert the tls error in the bundle.Status.
	if err != nil {
		log.Debugf("dial failed: %v", err)
		// record the error msg
		dialError = fmt.Sprintf("Failed rigid TLS handshake with %s: %v", dialName, err)
		// dial again with InsecureSkipVerify
		log.Debugf("try again with InsecureSkipVerify.")
		config.InsecureSkipVerify = true
		conn, err = tls.DialWithDialer(dialer, "tcp", dialName, config)
		if err != nil {
			log.Debugf("dial with InsecureSkipVerify failed: %v", err)
			return nil, errors.Wrap(errors.DialError, errors.Unknown, err)
		}
	}

	connState := conn.ConnectionState()

	certs := connState.PeerCertificates

	err = conn.VerifyHostname(serverName)
	if err != nil {
		log.Debugf("failed to verify hostname: %v", err)
		return nil, errors.Wrap(errors.CertificateError, errors.VerifyFailed, err)
	}

	// Bundle with remote certs. Inject the initial dial error, if any, to the status reporting.
	bundle, err := b.Bundle(certs, nil, flavor)
	if err != nil {
		return nil, err
	} else if dialError != "" {
		bundle.Status.Messages = append(bundle.Status.Messages, dialError)
	}
	return bundle, err
}

type fetchedIntermediate struct {
	Cert *x509.Certificate
	Name string
}

// fetchRemoteCertificate retrieves a single URL pointing to a certificate
// and attempts to first parse it as a DER-encoded certificate; if
// this fails, it attempts to decode it as a PEM-encoded certificate.
func fetchRemoteCertificate(certURL string) (fi *fetchedIntermediate, err error) {
	log.Debugf("fetching remote certificate: %s", certURL)
	var resp *http.Response
	resp, err = http.Get(certURL)
	if err != nil {
		log.Debugf("failed HTTP get: %v", err)
		return
	}

	defer resp.Body.Close()
	var certData []byte
	certData, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Debugf("failed to read response body: %v", err)
		return
	}

	log.Debugf("attempting to parse certificate as DER")
	crt, err := x509.ParseCertificate(certData)
	if err != nil {
		log.Debugf("attempting to parse certificate as PEM")
		crt, err = helpers.ParseCertificatePEM(certData)
		if err != nil {
			log.Debugf("failed to parse certificate: %v", err)
			return
		}
	}

	log.Debugf("certificate fetch succeeds")
	fi = &fetchedIntermediate{Cert: crt, Name: constructCertFileName(crt)}
	return
}

func reverse(certs []*x509.Certificate) []*x509.Certificate {
	n := len(certs)
	if n == 0 {
		return certs
	}
	rcerts := []*x509.Certificate{}
	for i := n - 1; i >= 0; i-- {
		rcerts = append(rcerts, certs[i])
	}
	return rcerts
}

// Check if the certs form a partial cert chain: every cert verifies
// the signature of the one in front of it.
func partialVerify(certs []*x509.Certificate) bool {
	n := len(certs)
	if n == 0 {
		return false
	}
	for i := 0; i < n-1; i++ {
		if certs[i].CheckSignatureFrom(certs[i+1]) != nil {
			return false
		}
	}
	return true
}

func isSelfSigned(cert *x509.Certificate) bool {
	return cert.CheckSignatureFrom(cert) == nil
}

func isChainRootNode(cert *x509.Certificate) bool {
	if isSelfSigned(cert) {
		return true
	}
	return false
}

func (b *Bundler) verifyChain(chain []*fetchedIntermediate) bool {
	// This process will verify if the root of the (partial) chain is in our root pool,
	// and will fail otherwise.
	log.Debugf("verifying chain")
	for vchain := chain[:]; len(vchain) > 0; vchain = vchain[1:] {
		cert := vchain[0]
		// If this is a certificate in one of the pools, skip it.
		if b.KnownIssuers[string(cert.Cert.Signature)] {
			log.Debugf("certificate is known")
			continue
		}

		_, err := cert.Cert.Verify(b.VerifyOptions())
		if err != nil {
			log.Debugf("certificate failed verification: %v", err)
			return false
		} else if len(chain) == len(vchain) && isChainRootNode(cert.Cert) {
			// The first certificate in the chain is a root; it shouldn't be stored.
			log.Debug("looking at root certificate, will not store")
			continue
		}

		// leaf cert has an empty name, don't store leaf cert.
		if cert.Name == "" {
			continue
		}

		log.Debug("add certificate to intermediate pool:", cert.Name)
		b.IntermediatePool.AddCert(cert.Cert)
		b.KnownIssuers[string(cert.Cert.Signature)] = true

		if IntermediateStash != "" {
			fileName := filepath.Join(IntermediateStash, cert.Name)

			var block = pem.Block{Type: "CERTIFICATE", Bytes: cert.Cert.Raw}

			log.Debugf("write intermediate to stash directory: %s", fileName)
			// If the write fails, verification should not fail.
			err = ioutil.WriteFile(fileName, pem.EncodeToMemory(&block), 0644)
			if err != nil {
				log.Errorf("failed to write new intermediate: %v", err)
			} else {
				log.Info("stashed new intermediate ", cert.Name)
			}
		}
	}
	return true
}

// constructCertFileName returns a uniquely identifying file name for a certificate
func constructCertFileName(cert *x509.Certificate) string {
	// construct the filename as the CN with no period and space
	name := strings.Replace(cert.Subject.CommonName, ".", "", -1)
	name = strings.Replace(name, " ", "", -1)

	// add SKI and serial number as extra identifier
	name += fmt.Sprintf("_%x", cert.SubjectKeyId)
	name += fmt.Sprintf("_%x", cert.SerialNumber.Bytes())

	name += ".crt"
	return name
}

// fetchIntermediates goes through each of the URLs in the AIA "Issuing
// CA" extensions and fetches those certificates. If those
// certificates are not present in either the root pool or
// intermediate pool, the certificate is saved to file and added to
// the list of intermediates to be used for verification. This will
// not add any new certificates to the root pool; if the ultimate
// issuer is not trusted, fetching the certicate here will not change
// that.
func (b *Bundler) fetchIntermediates(certs []*x509.Certificate) (err error) {
	if IntermediateStash != "" {
		log.Debugf("searching intermediates")
		if _, err := os.Stat(IntermediateStash); err != nil && os.IsNotExist(err) {
			log.Infof("intermediate stash directory %s doesn't exist, creating", IntermediateStash)
			err = os.MkdirAll(IntermediateStash, 0755)
			if err != nil {
				log.Errorf("failed to create intermediate stash directory %s: %v", IntermediateStash, err)
				return err
			}
			log.Infof("intermediate stash directory %s created", IntermediateStash)
		}
	}
	// stores URLs and certificate signatures that have been seen
	seen := map[string]bool{}
	var foundChains int

	// Construct a verify chain as a reversed partial bundle,
	// such that the certs are ordered by promxity to the root CAs.
	var chain []*fetchedIntermediate
	for i, cert := range certs {
		var name string

		// Only construct filenames for non-leaf intermediate certs
		// so they will be saved to disk if necessary.
		// Leaf cert gets a empty name and will be skipped.
		if i > 0 {
			name = constructCertFileName(cert)
		}

		chain = append([]*fetchedIntermediate{{cert, name}}, chain...)
		seen[string(cert.Signature)] = true
	}

	// Verify the chain and store valid intermediates in the chain.
	// If it doesn't verify, fetch the intermediates and extend the chain
	// in a DFS manner and verify each time we hit a root.
	for {
		if len(chain) == 0 {
			log.Debugf("search complete")
			if foundChains == 0 {
				return x509.UnknownAuthorityError{}
			}
			return nil
		}

		current := chain[0]
		var advanced bool
		if b.verifyChain(chain) {
			foundChains++
		}
		log.Debugf("walk AIA issuers")
		for _, url := range current.Cert.IssuingCertificateURL {
			if seen[url] {
				log.Debugf("url %s has been seen", url)
				continue
			}
			crt, err := fetchRemoteCertificate(url)
			if err != nil {
				continue
			} else if seen[string(crt.Cert.Signature)] {
				log.Debugf("fetched certificate is known")
				continue
			}
			seen[url] = true
			seen[string(crt.Cert.Signature)] = true
			chain = append([]*fetchedIntermediate{crt}, chain...)
			advanced = true
			break
		}

		if !advanced {
			log.Debugf("didn't advance, stepping back")
			chain = chain[1:]
		}
	}
}

// Bundle takes an X509 certificate (already in the
// Certificate structure), a private key as crypto.Signer in one of the appropriate
// formats (i.e. *rsa.PrivateKey or *ecdsa.PrivateKey, or even a opaque key), using them to
// build a certificate bundle.
func (b *Bundler) Bundle(certs []*x509.Certificate, key crypto.Signer, flavor BundleFlavor) (*Bundle, error) {
	log.Infof("bundling certificate for %+v", certs[0].Subject)
	if len(certs) == 0 {
		return nil, nil
	}

	// Detect reverse ordering of the cert chain.
	if len(certs) > 1 && !partialVerify(certs) {
		rcerts := reverse(certs)
		if partialVerify(rcerts) {
			certs = rcerts
		}
	}

	var ok bool
	cert := certs[0]
	if key != nil {
		switch {
		case cert.PublicKeyAlgorithm == x509.RSA:

			var rsaPublicKey *rsa.PublicKey
			if rsaPublicKey, ok = key.Public().(*rsa.PublicKey); !ok {
				return nil, errors.New(errors.PrivateKeyError, errors.KeyMismatch)
			}
			if cert.PublicKey.(*rsa.PublicKey).N.Cmp(rsaPublicKey.N) != 0 {
				return nil, errors.New(errors.PrivateKeyError, errors.KeyMismatch)
			}
		case cert.PublicKeyAlgorithm == x509.ECDSA:
			var ecdsaPublicKey *ecdsa.PublicKey
			if ecdsaPublicKey, ok = key.Public().(*ecdsa.PublicKey); !ok {
				return nil, errors.New(errors.PrivateKeyError, errors.KeyMismatch)
			}
			if cert.PublicKey.(*ecdsa.PublicKey).X.Cmp(ecdsaPublicKey.X) != 0 {
				return nil, errors.New(errors.PrivateKeyError, errors.KeyMismatch)
			}
		default:
			return nil, errors.New(errors.PrivateKeyError, errors.NotRSAOrECC)
		}
	} else {
		switch {
		case cert.PublicKeyAlgorithm == x509.RSA:
		case cert.PublicKeyAlgorithm == x509.ECDSA:
		default:
			return nil, errors.New(errors.PrivateKeyError, errors.NotRSAOrECC)
		}
	}

	bundle := new(Bundle)
	bundle.Cert = cert
	bundle.Key = key
	bundle.Issuer = &cert.Issuer
	bundle.Subject = &cert.Subject

	bundle.buildHostnames()

	if flavor == Force {
		// force bundle checks the certificates
		// forms a verification chain.
		if !partialVerify(certs) {
			return nil,
				errors.Wrap(errors.CertificateError, errors.VerifyFailed,
					goerr.New("Unable to verify the certificate chain"))
		}
		bundle.Chain = certs
	} else {
		// disallow self-signed cert
		if cert.CheckSignatureFrom(cert) == nil {
			return nil, errors.New(errors.CertificateError, errors.SelfSigned)
		}

		chains, err := cert.Verify(b.VerifyOptions())
		if err != nil {
			log.Debugf("verification failed: %v", err)
			// If the error was an unknown authority, try to fetch
			// the intermediate specified in the AIA and add it to
			// the intermediates bundle.
			if _, ok := err.(x509.UnknownAuthorityError); !ok {
				return nil, errors.Wrap(errors.CertificateError, errors.VerifyFailed, err)
			}

			log.Debugf("searching for intermediates via AIA issuer")
			searchErr := b.fetchIntermediates(certs)
			if searchErr != nil {
				log.Debugf("search failed: %v", searchErr)
				return nil, errors.Wrap(errors.CertificateError, errors.VerifyFailed, err)
			}

			log.Debugf("verifying new chain")
			chains, err = cert.Verify(b.VerifyOptions())
			if err != nil {
				log.Debugf("failed to verify chain: %v", err)
				return nil, errors.Wrap(errors.CertificateError, errors.VerifyFailed, err)
			}
			log.Debugf("verify ok")
		}
		var matchingChains [][]*x509.Certificate
		switch flavor {
		case Optimal:
			matchingChains = optimalChains(chains)
		case Ubiquitous:
			if len(ubiquity.Platforms) == 0 {
				log.Warning("No metadata, Ubiquitous falls back to Optimal.")
			}
			matchingChains = ubiquitousChains(chains)
		default:
			matchingChains = ubiquitousChains(chains)
		}

		bundle.Chain = matchingChains[0]
	}

	statusCode := int(errors.Success)
	var messages []string
	// Check if bundle is expiring.
	expiringCerts := checkExpiringCerts(bundle.Chain)
	if len(expiringCerts) > 0 {
		statusCode |= errors.BundleExpiringBit
		messages = append(messages, expirationWarning(expiringCerts))
	}
	// Check if bundle contains SHA2 certs.
	if ubiquity.ChainHashUbiquity(bundle.Chain) <= ubiquity.SHA2Ubiquity {
		statusCode |= errors.BundleNotUbiquitousBit
		messages = append(messages, sha2Warning)
	}
	// Check if bundle contains ECDSA signatures.
	if ubiquity.ChainKeyAlgoUbiquity(bundle.Chain) <= ubiquity.ECDSA256Ubiquity {
		statusCode |= errors.BundleNotUbiquitousBit
		messages = append(messages, ecdsaWarning)
	}

	// when forcing a bundle, bundle ubiquity doesn't matter
	// also we don't retrieve the anchoring root of the bundle
	var untrusted []string
	if flavor != Force {
		// Add root store presence info
		root := bundle.Chain[len(bundle.Chain)-1]
		bundle.Root = root
		log.Infof("the anchoring root is %v", root.Subject)
		// Check if there is any platform that doesn't trust the chain.
		// Also, an warning will be generated if ubiquity.Platforms is nil,
		untrusted = ubiquity.UntrustedPlatforms(root)
		untrustedMsg := untrustedPlatformsWarning(untrusted)
		if len(untrustedMsg) > 0 {
			log.Debug("Populate untrusted platform warning.")
			statusCode |= errors.BundleNotUbiquitousBit
			messages = append(messages, untrustedMsg)
		}
	}

	// Check if there is any platform that rejects the chain because of SHA1 deprecation.
	sha1Msgs := ubiquity.SHA1DeprecationMessages(bundle.Chain)
	if len(sha1Msgs) > 0 {
		log.Debug("Populate SHA1 deprecation warning.")
		statusCode |= errors.BundleNotUbiquitousBit
		messages = append(messages, sha1Msgs...)
	}

	bundle.Status = &BundleStatus{ExpiringSKIs: getSKIs(bundle.Chain, expiringCerts), Code: statusCode, Messages: messages, Untrusted: untrusted}

	// attempt to not to include the root certificate for optimization
	if flavor != Force {
		// Include at least one intermediate if the leaf has enabled OCSP and is not CA.
		if bundle.Cert.OCSPServer != nil && !bundle.Cert.IsCA && len(bundle.Chain) <= 2 {
			// No op. Return one intermediate if there is one.
		} else {
			// do not include the root.
			bundle.Chain = bundle.Chain[:len(bundle.Chain)-1]
		}
	}

	bundle.Status.IsRebundled = diff(bundle.Chain, certs)
	bundle.Expires = helpers.ExpiryTime(bundle.Chain)
	bundle.LeafExpires = bundle.Chain[0].NotAfter

	log.Debugf("bundle complete")
	return bundle, nil
}

// checkExpiringCerts returns indices of certs that are expiring within 30 days.
func checkExpiringCerts(chain []*x509.Certificate) (expiringIntermediates []int) {
	now := time.Now()
	for i, cert := range chain {
		if cert.NotAfter.Sub(now).Hours() < 720 {
			expiringIntermediates = append(expiringIntermediates, i)
		}
	}
	return
}

// getSKIs returns a list of cert subject key id  in the bundle chain with matched indices.
func getSKIs(chain []*x509.Certificate, indices []int) (skis []string) {
	for _, index := range indices {
		ski := fmt.Sprintf("%X", chain[index].SubjectKeyId)
		skis = append(skis, ski)
	}
	return
}

// expirationWarning generates a warning message with expiring certs.
func expirationWarning(expiringIntermediates []int) (ret string) {
	if len(expiringIntermediates) == 0 {
		return
	}

	ret = expiringWarningStub
	if len(expiringIntermediates) > 1 {
		ret = ret + "The expiring certs are"
	} else {
		ret = ret + "The expiring cert is"
	}
	for _, index := range expiringIntermediates {
		ret = ret + " #" + strconv.Itoa(index+1)
	}
	ret = ret + " in the chain."
	return
}

// untrustedPlatformsWarning generates a warning message with untrusted platform names.
func untrustedPlatformsWarning(platforms []string) string {
	if len(ubiquity.Platforms) == 0 {
		return ubiquityWarning
	}

	if len(platforms) == 0 {
		return ""
	}

	msg := untrustedWarningStub
	for i, platform := range platforms {
		if i > 0 {
			msg += ","
		}
		msg += " " + platform
	}
	msg += "."
	return msg
}

// Optimal chains are the shortest chains, with newest intermediates and most advanced crypto suite being the tie breaker.
func optimalChains(chains [][]*x509.Certificate) [][]*x509.Certificate {
	// Find shortest chains
	chains = ubiquity.Filter(chains, ubiquity.CompareChainLength)
	// Find the chains with longest expiry.
	chains = ubiquity.Filter(chains, ubiquity.CompareChainExpiry)
	// Find the chains with more advanced crypto suite
	chains = ubiquity.Filter(chains, ubiquity.CompareChainCryptoSuite)

	return chains
}

// Ubiquitous chains are the chains with highest platform coverage and break ties with the optimal strategy.
func ubiquitousChains(chains [][]*x509.Certificate) [][]*x509.Certificate {
	// Filter out chains with highest cross platform ubiquity.
	chains = ubiquity.Filter(chains, ubiquity.ComparePlatformUbiquity)
	// Prefer that all intermediates are SHA-2 certs if the leaf is a SHA-2 cert, in order to improve ubiquity.
	chains = ubiquity.Filter(chains, ubiquity.CompareSHA2Homogeneity)
	// Filter shortest chains
	chains = ubiquity.Filter(chains, ubiquity.CompareChainLength)
	// Filter chains with highest signature hash ubiquity.
	chains = ubiquity.Filter(chains, ubiquity.CompareChainHashUbiquity)
	// Filter chains with highest keyAlgo ubiquity.
	chains = ubiquity.Filter(chains, ubiquity.CompareChainKeyAlgoUbiquity)
	// Filter chains with intermediates that last longer.
	chains = ubiquity.Filter(chains, ubiquity.CompareExpiryUbiquity)
	// Use the optimal strategy as final tie breaker.
	return optimalChains(chains)
}

// diff checkes if two input cert chains are not identical
func diff(chain1, chain2 []*x509.Certificate) bool {
	// Check if bundled one is different from the input.
	diff := false
	if len(chain1) != len(chain2) {
		diff = true
	} else {
		for i := 0; i < len(chain1); i++ {
			cert1 := chain1[i]
			cert2 := chain2[i]
			// Use signature to differentiate.
			if !bytes.Equal(cert1.Signature, cert2.Signature) {
				diff = true
				break
			}
		}
	}
	return diff
}
