// Package config contains the configuration logic for CFSSL.
package config

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/asn1"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/auth"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	ocspConfig "github.com/cloudflare/cfssl/ocsp/config"
)

// A CSRWhitelist stores booleans for fields in the CSR. If a CSRWhitelist is
// not present in a SigningProfile, all of these fields may be copied from the
// CSR into the signed certificate. If a CSRWhitelist *is* present in a
// SigningProfile, only those fields with a `true` value in the CSRWhitelist may
// be copied from the CSR to the signed certificate. Note that some of these
// fields, like Subject, can be provided or partially provided through the API.
// Since API clients are expected to be trusted, but CSRs are not, fields
// provided through the API are not subject to whitelisting through this
// mechanism.
type CSRWhitelist struct {
	Subject, PublicKeyAlgorithm, PublicKey, SignatureAlgorithm bool
	DNSNames, IPAddresses, EmailAddresses                      bool
}

// OID is our own version of asn1's ObjectIdentifier, so we can define a custom
// JSON marshal / unmarshal.
type OID asn1.ObjectIdentifier

// CertificatePolicy represents the ASN.1 PolicyInformation structure from
// https://tools.ietf.org/html/rfc3280.html#page-106.
// Valid values of Type are "id-qt-unotice" and "id-qt-cps"
type CertificatePolicy struct {
	ID         OID
	Qualifiers []CertificatePolicyQualifier
}

// CertificatePolicyQualifier represents a single qualifier from an ASN.1
// PolicyInformation structure.
type CertificatePolicyQualifier struct {
	Type  string
	Value string
}

// AuthRemote is an authenticated remote signer.
type AuthRemote struct {
	RemoteName  string `json:"remote"`
	AuthKeyName string `json:"auth_key"`
}

// CAConstraint specifies various CA constraints on the signed certificate.
// CAConstraint would verify against (and override) the CA
// extensions in the given CSR.
type CAConstraint struct {
	IsCA           bool `json:"is_ca"`
	MaxPathLen     int  `json:"max_path_len"`
	MaxPathLenZero bool `json:"max_path_len_zero"`
}

// A SigningProfile stores information that the CA needs to store
// signature policy.
type SigningProfile struct {
	Usage               []string     `json:"usages"`
	IssuerURL           []string     `json:"issuer_urls"`
	OCSP                string       `json:"ocsp_url"`
	CRL                 string       `json:"crl_url"`
	CAConstraint        CAConstraint `json:"ca_constraint"`
	OCSPNoCheck         bool         `json:"ocsp_no_check"`
	ExpiryString        string       `json:"expiry"`
	BackdateString      string       `json:"backdate"`
	AuthKeyName         string       `json:"auth_key"`
	RemoteName          string       `json:"remote"`
	NotBefore           time.Time    `json:"not_before"`
	NotAfter            time.Time    `json:"not_after"`
	NameWhitelistString string       `json:"name_whitelist"`
	AuthRemote          AuthRemote   `json:"auth_remote"`
	CTLogServers        []string     `json:"ct_log_servers"`
	AllowedExtensions   []OID        `json:"allowed_extensions"`
	CertStore           string       `json:"cert_store"`

	Policies                    []CertificatePolicy
	Expiry                      time.Duration
	Backdate                    time.Duration
	Provider                    auth.Provider
	RemoteProvider              auth.Provider
	RemoteServer                string
	RemoteCAs                   *x509.CertPool
	ClientCert                  *tls.Certificate
	CSRWhitelist                *CSRWhitelist
	NameWhitelist               *regexp.Regexp
	ExtensionWhitelist          map[string]bool
	ClientProvidesSerialNumbers bool
}

// UnmarshalJSON unmarshals a JSON string into an OID.
func (oid *OID) UnmarshalJSON(data []byte) (err error) {
	if data[0] != '"' || data[len(data)-1] != '"' {
		return errors.New("OID JSON string not wrapped in quotes." + string(data))
	}
	data = data[1 : len(data)-1]
	parsedOid, err := parseObjectIdentifier(string(data))
	if err != nil {
		return err
	}
	*oid = OID(parsedOid)
	return
}

// MarshalJSON marshals an oid into a JSON string.
func (oid OID) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"%v"`, asn1.ObjectIdentifier(oid))), nil
}

func parseObjectIdentifier(oidString string) (oid asn1.ObjectIdentifier, err error) {
	validOID, err := regexp.MatchString("\\d(\\.\\d+)*", oidString)
	if err != nil {
		return
	}
	if !validOID {
		err = errors.New("Invalid OID")
		return
	}

	segments := strings.Split(oidString, ".")
	oid = make(asn1.ObjectIdentifier, len(segments))
	for i, intString := range segments {
		oid[i], err = strconv.Atoi(intString)
		if err != nil {
			return
		}
	}
	return
}

const timeFormat = "2006-01-02T15:04:05"

// populate is used to fill in the fields that are not in JSON
//
// First, the ExpiryString parameter is needed to parse
// expiration timestamps from JSON. The JSON decoder is not able to
// decode a string time duration to a time.Duration, so this is called
// when loading the configuration to properly parse and fill out the
// Expiry parameter.
// This function is also used to create references to the auth key
// and default remote for the profile.
// It returns true if ExpiryString is a valid representation of a
// time.Duration, and the AuthKeyString and RemoteName point to
// valid objects. It returns false otherwise.
func (p *SigningProfile) populate(cfg *Config) error {
	if p == nil {
		return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, errors.New("can't parse nil profile"))
	}

	var err error
	if p.RemoteName == "" && p.AuthRemote.RemoteName == "" {
		log.Debugf("parse expiry in profile")
		if p.ExpiryString == "" {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, errors.New("empty expiry string"))
		}

		dur, err := time.ParseDuration(p.ExpiryString)
		if err != nil {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, err)
		}

		log.Debugf("expiry is valid")
		p.Expiry = dur

		if p.BackdateString != "" {
			dur, err = time.ParseDuration(p.BackdateString)
			if err != nil {
				return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, err)
			}

			p.Backdate = dur
		}

		if !p.NotBefore.IsZero() && !p.NotAfter.IsZero() && p.NotAfter.Before(p.NotBefore) {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, err)
		}

		if len(p.Policies) > 0 {
			for _, policy := range p.Policies {
				for _, qualifier := range policy.Qualifiers {
					if qualifier.Type != "" && qualifier.Type != "id-qt-unotice" && qualifier.Type != "id-qt-cps" {
						return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
							errors.New("invalid policy qualifier type"))
					}
				}
			}
		}
	} else if p.RemoteName != "" {
		log.Debug("match remote in profile to remotes section")
		if p.AuthRemote.RemoteName != "" {
			log.Error("profile has both a remote and an auth remote specified")
			return cferr.New(cferr.PolicyError, cferr.InvalidPolicy)
		}
		if remote := cfg.Remotes[p.RemoteName]; remote != "" {
			if err := p.updateRemote(remote); err != nil {
				return err
			}
		} else {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
				errors.New("failed to find remote in remotes section"))
		}
	} else {
		log.Debug("match auth remote in profile to remotes section")
		if remote := cfg.Remotes[p.AuthRemote.RemoteName]; remote != "" {
			if err := p.updateRemote(remote); err != nil {
				return err
			}
		} else {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
				errors.New("failed to find remote in remotes section"))
		}
	}

	if p.AuthKeyName != "" {
		log.Debug("match auth key in profile to auth_keys section")
		if key, ok := cfg.AuthKeys[p.AuthKeyName]; ok == true {
			if key.Type == "standard" {
				p.Provider, err = auth.New(key.Key, nil)
				if err != nil {
					log.Debugf("failed to create new standard auth provider: %v", err)
					return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
						errors.New("failed to create new standard auth provider"))
				}
			} else {
				log.Debugf("unknown authentication type %v", key.Type)
				return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
					errors.New("unknown authentication type"))
			}
		} else {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
				errors.New("failed to find auth_key in auth_keys section"))
		}
	}

	if p.AuthRemote.AuthKeyName != "" {
		log.Debug("match auth remote key in profile to auth_keys section")
		if key, ok := cfg.AuthKeys[p.AuthRemote.AuthKeyName]; ok == true {
			if key.Type == "standard" {
				p.RemoteProvider, err = auth.New(key.Key, nil)
				if err != nil {
					log.Debugf("failed to create new standard auth provider: %v", err)
					return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
						errors.New("failed to create new standard auth provider"))
				}
			} else {
				log.Debugf("unknown authentication type %v", key.Type)
				return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
					errors.New("unknown authentication type"))
			}
		} else {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
				errors.New("failed to find auth_remote's auth_key in auth_keys section"))
		}
	}

	if p.NameWhitelistString != "" {
		log.Debug("compiling whitelist regular expression")
		rule, err := regexp.Compile(p.NameWhitelistString)
		if err != nil {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
				errors.New("failed to compile name whitelist section"))
		}
		p.NameWhitelist = rule
	}

	p.ExtensionWhitelist = map[string]bool{}
	for _, oid := range p.AllowedExtensions {
		p.ExtensionWhitelist[asn1.ObjectIdentifier(oid).String()] = true
	}

	return nil
}

// updateRemote takes a signing profile and initializes the remote server object
// to the hostname:port combination sent by remote.
func (p *SigningProfile) updateRemote(remote string) error {
	if remote != "" {
		p.RemoteServer = remote
	}
	return nil
}

// OverrideRemotes takes a signing configuration and updates the remote server object
// to the hostname:port combination sent by remote
func (p *Signing) OverrideRemotes(remote string) error {
	if remote != "" {
		var err error
		for _, profile := range p.Profiles {
			err = profile.updateRemote(remote)
			if err != nil {
				return err
			}
		}
		err = p.Default.updateRemote(remote)
		if err != nil {
			return err
		}
	}
	return nil
}

// SetClientCertKeyPairFromFile updates the properties to set client certificates for mutual
// authenticated TLS remote requests
func (p *Signing) SetClientCertKeyPairFromFile(certFile string, keyFile string) error {
	if certFile != "" && keyFile != "" {
		cert, err := helpers.LoadClientCertificate(certFile, keyFile)
		if err != nil {
			return err
		}
		for _, profile := range p.Profiles {
			profile.ClientCert = cert
		}
		p.Default.ClientCert = cert
	}
	return nil
}

// SetRemoteCAsFromFile reads root CAs from file and updates the properties to set remote CAs for TLS
// remote requests
func (p *Signing) SetRemoteCAsFromFile(caFile string) error {
	if caFile != "" {
		remoteCAs, err := helpers.LoadPEMCertPool(caFile)
		if err != nil {
			return err
		}
		p.SetRemoteCAs(remoteCAs)
	}
	return nil
}

// SetRemoteCAs updates the properties to set remote CAs for TLS
// remote requests
func (p *Signing) SetRemoteCAs(remoteCAs *x509.CertPool) {
	for _, profile := range p.Profiles {
		profile.RemoteCAs = remoteCAs
	}
	p.Default.RemoteCAs = remoteCAs
}

// NeedsRemoteSigner returns true if one of the profiles has a remote set
func (p *Signing) NeedsRemoteSigner() bool {
	for _, profile := range p.Profiles {
		if profile.RemoteServer != "" {
			return true
		}
	}

	if p.Default.RemoteServer != "" {
		return true
	}

	return false
}

// NeedsLocalSigner returns true if one of the profiles doe not have a remote set
func (p *Signing) NeedsLocalSigner() bool {
	for _, profile := range p.Profiles {
		if profile.RemoteServer == "" {
			return true
		}
	}

	if p.Default.RemoteServer == "" {
		return true
	}

	return false
}

// Usages parses the list of key uses in the profile, translating them
// to a list of X.509 key usages and extended key usages.  The unknown
// uses are collected into a slice that is also returned.
func (p *SigningProfile) Usages() (ku x509.KeyUsage, eku []x509.ExtKeyUsage, unk []string) {
	for _, keyUse := range p.Usage {
		if kuse, ok := KeyUsage[keyUse]; ok {
			ku |= kuse
		} else if ekuse, ok := ExtKeyUsage[keyUse]; ok {
			eku = append(eku, ekuse)
		} else {
			unk = append(unk, keyUse)
		}
	}
	return
}

// A valid profile must be a valid local profile or a valid remote profile.
// A valid local profile has defined at least key usages to be used, and a
// valid local default profile has defined at least a default expiration.
// A valid remote profile (default or not) has remote signer initialized.
// In addition, a remote profile must has a valid auth provider if auth
// key defined.
func (p *SigningProfile) validProfile(isDefault bool) bool {
	if p == nil {
		return false
	}

	if p.AuthRemote.RemoteName == "" && p.AuthRemote.AuthKeyName != "" {
		log.Debugf("invalid auth remote profile: no remote signer specified")
		return false
	}

	if p.RemoteName != "" {
		log.Debugf("validate remote profile")

		if p.RemoteServer == "" {
			log.Debugf("invalid remote profile: no remote signer specified")
			return false
		}

		if p.AuthKeyName != "" && p.Provider == nil {
			log.Debugf("invalid remote profile: auth key name is defined but no auth provider is set")
			return false
		}

		if p.AuthRemote.RemoteName != "" {
			log.Debugf("invalid remote profile: auth remote is also specified")
			return false
		}
	} else if p.AuthRemote.RemoteName != "" {
		log.Debugf("validate auth remote profile")
		if p.RemoteServer == "" {
			log.Debugf("invalid auth remote profile: no remote signer specified")
			return false
		}

		if p.AuthRemote.AuthKeyName == "" || p.RemoteProvider == nil {
			log.Debugf("invalid auth remote profile: no auth key is defined")
			return false
		}
	} else {
		log.Debugf("validate local profile")
		if !isDefault {
			if len(p.Usage) == 0 {
				log.Debugf("invalid local profile: no usages specified")
				return false
			} else if _, _, unk := p.Usages(); len(unk) == len(p.Usage) {
				log.Debugf("invalid local profile: no valid usages")
				return false
			}
		} else {
			if p.Expiry == 0 {
				log.Debugf("invalid local profile: no expiry set")
				return false
			}
		}
	}

	log.Debugf("profile is valid")
	return true
}

// This checks if the SigningProfile object contains configurations that are only effective with a local signer
// which has access to CA private key.
func (p *SigningProfile) hasLocalConfig() bool {
	if p.Usage != nil ||
		p.IssuerURL != nil ||
		p.OCSP != "" ||
		p.ExpiryString != "" ||
		p.BackdateString != "" ||
		p.CAConstraint.IsCA != false ||
		!p.NotBefore.IsZero() ||
		!p.NotAfter.IsZero() ||
		p.NameWhitelistString != "" ||
		len(p.CTLogServers) != 0 {
		return true
	}
	return false
}

// warnSkippedSettings prints a log warning message about skipped settings
// in a SigningProfile, usually due to remote signer.
func (p *Signing) warnSkippedSettings() {
	const warningMessage = `The configuration value by "usages", "issuer_urls", "ocsp_url", "crl_url", "ca_constraint", "expiry", "backdate", "not_before", "not_after", "cert_store" and "ct_log_servers" are skipped`
	if p == nil {
		return
	}

	if (p.Default.RemoteName != "" || p.Default.AuthRemote.RemoteName != "") && p.Default.hasLocalConfig() {
		log.Warning("default profile points to a remote signer: ", warningMessage)
	}

	for name, profile := range p.Profiles {
		if (profile.RemoteName != "" || profile.AuthRemote.RemoteName != "") && profile.hasLocalConfig() {
			log.Warningf("Profiles[%s] points to a remote signer: %s", name, warningMessage)
		}
	}
}

// Signing codifies the signature configuration policy for a CA.
type Signing struct {
	Profiles map[string]*SigningProfile `json:"profiles"`
	Default  *SigningProfile            `json:"default"`
}

// Config stores configuration information for the CA.
type Config struct {
	Signing  *Signing           `json:"signing"`
	OCSP     *ocspConfig.Config `json:"ocsp"`
	AuthKeys map[string]AuthKey `json:"auth_keys,omitempty"`
	Remotes  map[string]string  `json:"remotes,omitempty"`
}

// Valid ensures that Config is a valid configuration. It should be
// called immediately after parsing a configuration file.
func (c *Config) Valid() bool {
	return c.Signing.Valid()
}

// Valid checks the signature policies, ensuring they are valid
// policies. A policy is valid if it has defined at least key usages
// to be used, and a valid default profile has defined at least a
// default expiration.
func (p *Signing) Valid() bool {
	if p == nil {
		return false
	}

	log.Debugf("validating configuration")
	if !p.Default.validProfile(true) {
		log.Debugf("default profile is invalid")
		return false
	}

	for _, sp := range p.Profiles {
		if !sp.validProfile(false) {
			log.Debugf("invalid profile")
			return false
		}
	}

	p.warnSkippedSettings()

	return true
}

// KeyUsage contains a mapping of string names to key usages.
var KeyUsage = map[string]x509.KeyUsage{
	"signing":             x509.KeyUsageDigitalSignature,
	"digital signature":   x509.KeyUsageDigitalSignature,
	"content committment": x509.KeyUsageContentCommitment,
	"key encipherment":    x509.KeyUsageKeyEncipherment,
	"key agreement":       x509.KeyUsageKeyAgreement,
	"data encipherment":   x509.KeyUsageDataEncipherment,
	"cert sign":           x509.KeyUsageCertSign,
	"crl sign":            x509.KeyUsageCRLSign,
	"encipher only":       x509.KeyUsageEncipherOnly,
	"decipher only":       x509.KeyUsageDecipherOnly,
}

// ExtKeyUsage contains a mapping of string names to extended key
// usages.
var ExtKeyUsage = map[string]x509.ExtKeyUsage{
	"any":              x509.ExtKeyUsageAny,
	"server auth":      x509.ExtKeyUsageServerAuth,
	"client auth":      x509.ExtKeyUsageClientAuth,
	"code signing":     x509.ExtKeyUsageCodeSigning,
	"email protection": x509.ExtKeyUsageEmailProtection,
	"s/mime":           x509.ExtKeyUsageEmailProtection,
	"ipsec end system": x509.ExtKeyUsageIPSECEndSystem,
	"ipsec tunnel":     x509.ExtKeyUsageIPSECTunnel,
	"ipsec user":       x509.ExtKeyUsageIPSECUser,
	"timestamping":     x509.ExtKeyUsageTimeStamping,
	"ocsp signing":     x509.ExtKeyUsageOCSPSigning,
	"microsoft sgc":    x509.ExtKeyUsageMicrosoftServerGatedCrypto,
	"netscape sgc":     x509.ExtKeyUsageNetscapeServerGatedCrypto,
}

// An AuthKey contains an entry for a key used for authentication.
type AuthKey struct {
	// Type contains information needed to select the appropriate
	// constructor. For example, "standard" for HMAC-SHA-256,
	// "standard-ip" for HMAC-SHA-256 incorporating the client's
	// IP.
	Type string `json:"type"`
	// Key contains the key information, such as a hex-encoded
	// HMAC key.
	Key string `json:"key"`
}

// DefaultConfig returns a default configuration specifying basic key
// usage and a 1 year expiration time. The key usages chosen are
// signing, key encipherment, client auth and server auth.
func DefaultConfig() *SigningProfile {
	d := helpers.OneYear
	return &SigningProfile{
		Usage:        []string{"signing", "key encipherment", "server auth", "client auth"},
		Expiry:       d,
		ExpiryString: "8760h",
	}
}

// LoadFile attempts to load the configuration file stored at the path
// and returns the configuration. On error, it returns nil.
func LoadFile(path string) (*Config, error) {
	log.Debugf("loading configuration file from %s", path)
	if path == "" {
		return nil, cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, errors.New("invalid path"))
	}

	body, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, errors.New("could not read configuration file"))
	}

	return LoadConfig(body)
}

// LoadConfig attempts to load the configuration from a byte slice.
// On error, it returns nil.
func LoadConfig(config []byte) (*Config, error) {
	var cfg = &Config{}
	err := json.Unmarshal(config, &cfg)
	if err != nil {
		return nil, cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy,
			errors.New("failed to unmarshal configuration: "+err.Error()))
	}

	if cfg.Signing == nil {
		return nil, errors.New("No \"signing\" field present")
	}

	if cfg.Signing.Default == nil {
		log.Debugf("no default given: using default config")
		cfg.Signing.Default = DefaultConfig()
	} else {
		if err := cfg.Signing.Default.populate(cfg); err != nil {
			return nil, err
		}
	}

	for k := range cfg.Signing.Profiles {
		if err := cfg.Signing.Profiles[k].populate(cfg); err != nil {
			return nil, err
		}
	}

	if !cfg.Valid() {
		return nil, cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, errors.New("invalid configuration"))
	}

	log.Debugf("configuration ok")
	return cfg, nil
}
