package ca

import (
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"net"
	"path/filepath"

	"github.com/cloudflare/cfssl/api/client"
	"github.com/cloudflare/cfssl/auth"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/transport/core"
)

type authError struct {
	authType string
}

func (err *authError) Error() string {
	return fmt.Sprintf("transport: unsupported CFSSL authentication method %s", err.authType)
}

// This approach allows us to quickly add other providers later, such
// as the TPM.
var authTypes = map[string]func(config.AuthKey, []byte) (auth.Provider, error){
	"standard": newStandardProvider,
}

// Create a standard provider without providing any additional data.
func newStandardProvider(ak config.AuthKey, ad []byte) (auth.Provider, error) {
	return auth.New(ak.Key, ad)
}

// Create a new provider from an authentication key and possibly
// additional data.
func newProvider(ak config.AuthKey, ad []byte) (auth.Provider, error) {
	// If no auth key was provided, use unauthenticated
	// requests. This is useful when a local CFSSL is being used.
	if ak.Type == "" && ak.Key == "" {
		return nil, nil
	}

	f, ok := authTypes[ak.Type]
	if !ok {
		return nil, &authError{authType: ak.Type}
	}

	return f(ak, ad)
}

// ErrNoAuth is returned when a client is talking to a CFSSL remote
// that is not on a loopback address and doesn't have an
// authentication provider set.
var ErrNoAuth = errors.New("transport: authentication is required for non-local remotes")

var v4Loopback = net.IPNet{
	IP:   net.IP{127, 0, 0, 0},
	Mask: net.IPv4Mask(255, 0, 0, 0),
}

func ipIsLocal(ip net.IP) bool {
	if ip.To4() == nil {
		return ip.Equal(net.IPv6loopback)
	}

	return v4Loopback.Contains(ip)
}

// The only time a client should be doing unauthenticated requests is
// when a local CFSSL is being used.
func (cap *CFSSL) validateAuth() error {
	// The client is using some form of authentication, and the best way
	// to figure out that the auth is invalid is when it's used. Therefore,
	// we'll delay checking the credentials until that time.
	if cap.provider != nil {
		return nil
	}

	hosts := cap.remote.Hosts()
	for i := range hosts {
		ips, err := net.LookupIP(hosts[i])
		if err != nil {
			return err
		}

		for _, ip := range ips {
			if !ipIsLocal(ip) {
				return ErrNoAuth
			}
		}
	}

	return nil
}

var cfsslConfigDirs = []string{
	"/usr/local/cfssl",
	"/etc/cfssl",
	"/state/etc/cfssl",
}

// The CFSSL standard is to have a configuration file for a label as
// <config>.json.
func findLabel(label string) *config.Config {
	for _, dir := range cfsslConfigDirs {
		cfgFile := filepath.Join(dir, label+".json")
		cfg, err := config.LoadFile(cfgFile)
		if err == nil {
			return cfg
		}
	}

	return nil
}

func getProfile(cfg *config.Config, profileName string) (*config.SigningProfile, bool) {
	if cfg == nil || cfg.Signing == nil || cfg.Signing.Default == nil {
		return nil, false
	}

	var ok bool
	profile := cfg.Signing.Default
	if profileName != "" {
		if cfg.Signing.Profiles == nil {
			return nil, false
		}

		profile, ok = cfg.Signing.Profiles[profileName]
		if !ok {
			return nil, false
		}
	}

	return profile, true
}

// loadAuth loads an authentication provider from the client config's
// explicitly set auth key.
func (cap *CFSSL) loadAuth() error {
	var err error
	cap.provider, err = newProvider(cap.DefaultAuth, nil)
	return err
}

func getRemote(cfg *config.Config, profile *config.SigningProfile) (string, bool) {
	// NB: Loading the config will validate that the remote is
	// present in the config's remote section.
	if profile.RemoteServer != "" {
		return profile.RemoteServer, true
	}

	return "", false
}

func (cap *CFSSL) setRemoteAndAuth() error {
	if cap.Label != "" {
		cfsslConfig := findLabel(cap.Label)
		profile, ok := getProfile(cfsslConfig, cap.Profile)
		if ok {
			remote, ok := getRemote(cfsslConfig, profile)
			if ok {
				cap.remote = client.NewServer(remote)
				cap.provider = profile.Provider
				return nil
			}

			// The profile may not have a remote set, but
			// it may have an authentication provider.
			cap.provider = profile.Provider
		}
	}

	cap.remote = cap.DefaultRemote
	if cap.provider != nil {
		return nil
	}
	return cap.loadAuth()
}

// CFSSL provides support for signing certificates via
// CFSSL.
type CFSSL struct {
	remote        client.Remote
	provider      auth.Provider
	Profile       string
	Label         string
	DefaultRemote client.Remote
	DefaultAuth   config.AuthKey
}

// SignCSR requests a certificate from a CFSSL signer.
func (cap *CFSSL) SignCSR(csrPEM []byte) (cert []byte, err error) {
	p, _ := pem.Decode(csrPEM)
	if p == nil || p.Type != "CERTIFICATE REQUEST" {
		return nil, errors.New("transport: invalid PEM-encoded certificate signing request")
	}

	csr, err := x509.ParseCertificateRequest(p.Bytes)
	if err != nil {
		return nil, err
	}

	hosts := make([]string, 0, len(csr.DNSNames)+len(csr.IPAddresses))
	copy(hosts, csr.DNSNames)

	for i := range csr.IPAddresses {
		hosts = append(hosts, csr.IPAddresses[i].String())
	}

	sreq := &signer.SignRequest{
		Hosts:   hosts,
		Request: string(csrPEM),
		Profile: cap.Profile,
		Label:   cap.Label,
	}

	out, err := json.Marshal(sreq)
	if err != nil {
		return nil, err
	}

	if cap.provider != nil {
		return cap.remote.AuthSign(out, nil, cap.provider)
	}

	return cap.remote.Sign(out)
}

// CACertificate returns the certificate for a CFSSL CA.
func (cap *CFSSL) CACertificate() ([]byte, error) {
	req := &info.Req{
		Label:   cap.Label,
		Profile: cap.Profile,
	}
	out, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := cap.remote.Info(out)
	if err != nil {
		return nil, err
	}

	return []byte(resp.Certificate), nil
}

// NewCFSSLProvider takes the configuration information from an
// Identity (and an optional default remote), returning a CFSSL
// instance. There should be a profile in id called "cfssl", which
// should contain label and profile fields as needed.
func NewCFSSLProvider(id *core.Identity, defaultRemote client.Remote) (*CFSSL, error) {
	if id == nil {
		return nil, errors.New("transport: the identity hasn't been initialised. Has it been loaded from disk?")
	}

	cap := &CFSSL{
		DefaultRemote: defaultRemote,
	}

	cfssl := id.Profiles["cfssl"]
	if cfssl != nil {
		cap.Label = cfssl["label"]
		cap.Profile = cfssl["profile"]

		if cap.DefaultRemote == nil {
			cap.DefaultRemote = client.NewServer(cfssl["remote"])
		}

		cap.DefaultAuth.Type = cfssl["auth-type"]
		cap.DefaultAuth.Key = cfssl["auth-key"]
	}

	err := cap.setRemoteAndAuth()
	if err != nil {
		return nil, err
	}

	return cap, nil
}
