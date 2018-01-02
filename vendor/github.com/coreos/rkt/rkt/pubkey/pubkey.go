// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pubkey

import (
	"bufio"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/coreos/rkt/pkg/keystore"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/rkt/config"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/discovery"
	"golang.org/x/crypto/openpgp"
	"golang.org/x/crypto/ssh/terminal"
)

type Manager struct {
	AuthPerHost          map[string]config.Headerer
	InsecureAllowHTTP    bool
	InsecureSkipTLSCheck bool
	TrustKeysFromHTTPS   bool
	Ks                   *keystore.Keystore
	Debug                bool
}

type AcceptOption int

const (
	AcceptForce AcceptOption = iota
	AcceptAsk
)

var (
	log    *rktlog.Logger
	stdout *rktlog.Logger = rktlog.New(os.Stdout, "", false)

	secureClient   = newClient(false)
	insecureClient = newClient(true)
)

func ensureLogger(debug bool) {
	if log == nil {
		log = rktlog.New(os.Stderr, "pubkey", debug)
	}
}

// GetPubKeyLocations discovers locations at prefix
func (m *Manager) GetPubKeyLocations(prefix string) ([]string, error) {
	ensureLogger(m.Debug)
	if prefix == "" {
		return nil, fmt.Errorf("empty prefix")
	}

	kls, err := m.metaDiscoverPubKeyLocations(prefix)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("prefix meta discovery error"), err)
	}

	if len(kls) == 0 {
		return nil, fmt.Errorf("meta discovery on %s resulted in no keys", prefix)
	}

	return kls, nil
}

// AddKeys adds the keys listed in pkls at prefix
func (m *Manager) AddKeys(pkls []string, prefix string, accept AcceptOption) error {
	ensureLogger(m.Debug)
	if m.Ks == nil {
		return fmt.Errorf("no keystore available to add keys to")
	}

	for _, pkl := range pkls {
		u, err := url.Parse(pkl)
		if err != nil {
			return err
		}
		pk, err := m.getPubKey(u)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("error accessing the key %s", pkl), err)
		}
		defer pk.Close()

		err = displayKey(prefix, pkl, pk)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("error displaying the key %s", pkl), err)
		}

		if m.TrustKeysFromHTTPS && u.Scheme == "https" {
			accept = AcceptForce
		}

		if accept == AcceptAsk {
			if !terminal.IsTerminal(int(os.Stdin.Fd())) || !terminal.IsTerminal(int(os.Stderr.Fd())) {
				log.Printf("To trust the key for %q, do one of the following:", prefix)
				log.Printf(" - call rkt with --trust-keys-from-https")
				log.Printf(" - run: rkt trust --prefix %q", prefix)
				return fmt.Errorf("error reviewing key: unable to ask user to review fingerprint due to lack of tty")
			}
			accepted, err := reviewKey()
			if err != nil {
				return errwrap.Wrap(errors.New("error reviewing key"), err)
			}
			if !accepted {
				log.Printf("not trusting %q", pkl)
				continue
			}
		}

		if accept == AcceptForce {
			stdout.Printf("Trusting %q for prefix %q without fingerprint review.", pkl, prefix)
		} else {
			stdout.Printf("Trusting %q for prefix %q after fingerprint review.", pkl, prefix)
		}

		if prefix == "" {
			path, err := m.Ks.StoreTrustedKeyRoot(pk)
			if err != nil {
				return errwrap.Wrap(errors.New("error adding root key"), err)
			}
			stdout.Printf("Added root key at %q", path)
		} else {
			path, err := m.Ks.StoreTrustedKeyPrefix(prefix, pk)
			if err != nil {
				return errwrap.Wrap(fmt.Errorf("error adding key for prefix %q", prefix), err)
			}
			stdout.Printf("Added key for prefix %q at %q", prefix, path)
		}
	}
	return nil
}

// metaDiscoverPubKeyLocations discovers the locations of public keys through ACDiscovery by applying prefix as an ACApp
func (m *Manager) metaDiscoverPubKeyLocations(prefix string) ([]string, error) {
	app, err := discovery.NewAppFromString(prefix)
	if err != nil {
		return nil, err
	}

	hostHeaders := config.ResolveAuthPerHost(m.AuthPerHost)
	insecure := discovery.InsecureNone

	if m.InsecureAllowHTTP {
		insecure = insecure | discovery.InsecureHTTP
	}
	if m.InsecureSkipTLSCheck {
		insecure = insecure | discovery.InsecureTLS
	}
	keys, attempts, err := discovery.DiscoverPublicKeys(*app, hostHeaders, insecure, 0)
	if err != nil {
		return nil, err
	}

	if m.Debug {
		for _, a := range attempts {
			log.PrintE(fmt.Sprintf("meta tag 'ac-discovery-pubkeys' not found on %s", a.Prefix), a.Error)
		}
	}

	return keys, nil
}

// getPubKey retrieves a public key (if remote), and verifies it's a gpg key
func (m *Manager) getPubKey(u *url.URL) (*os.File, error) {
	switch u.Scheme {
	case "":
		return os.Open(u.Path)
	case "http":
		if !m.InsecureAllowHTTP {
			return nil, fmt.Errorf("--insecure-allow-http required for http URLs")
		}
		fallthrough
	case "https":
		return downloadKey(u, m.InsecureSkipTLSCheck)
	}

	return nil, fmt.Errorf("only local files and http or https URLs supported")
}

// downloadKey retrieves the file, storing it in a deleted tempfile
func downloadKey(u *url.URL, skipTLSCheck bool) (*os.File, error) {
	tf, err := ioutil.TempFile("", "")
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error creating tempfile"), err)
	}
	os.Remove(tf.Name()) // no need to keep the tempfile around

	defer func() {
		if tf != nil {
			tf.Close()
		}
	}()

	// TODO(krnowak): we should probably apply credential headers
	// from config here
	var client *http.Client
	if skipTLSCheck {
		client = insecureClient
	} else {
		client = secureClient
	}

	res, err := client.Get(u.String())
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error getting key"), err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bad HTTP status code: %d", res.StatusCode)
	}

	if _, err := io.Copy(tf, res.Body); err != nil {
		return nil, errwrap.Wrap(errors.New("error copying key"), err)
	}

	if _, err = tf.Seek(0, os.SEEK_SET); err != nil {
		return nil, errwrap.Wrap(errors.New("error seeking"), err)
	}

	retTf := tf
	tf = nil
	return retTf, nil
}

func newClient(skipTLSCheck bool) *http.Client {
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	} // values taken from stdlib v1.5.3

	tr := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		Dial:                dialer.Dial,
		TLSHandshakeTimeout: 10 * time.Second,
	} // values taken from stdlib v1.5.3

	if skipTLSCheck {
		tr.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: true,
		}
	}

	return &http.Client{
		Transport: tr,

		// keys are rather small, long download times are not expected, hence setting a client timeout.
		// firefox uses a 30s read timeout, being pessimistic bump to 2 minutes.
		// Note that this includes connection, and tls handshake times, see https://blog.cloudflare.com/the-complete-guide-to-golang-net-http-timeouts/#clienttimeouts
		Timeout: 2 * time.Minute,
	}
}

// displayKey shows the key summary
func displayKey(prefix, location string, key *os.File) error {
	defer key.Seek(0, os.SEEK_SET)

	kr, err := openpgp.ReadArmoredKeyRing(key)
	if err != nil {
		return errwrap.Wrap(errors.New("error reading key"), err)
	}

	log.Printf("prefix: %q\nkey: %q", prefix, location)
	for _, k := range kr {
		stdout.Printf("gpg key fingerprint is: %s", fingerToString(k.PrimaryKey.Fingerprint))
		for _, sk := range k.Subkeys {
			stdout.Printf("    Subkey fingerprint: %s", fingerToString(sk.PublicKey.Fingerprint))
		}

		for n := range k.Identities {
			stdout.Printf("\t%s", n)
		}
	}

	return nil
}

// reviewKey asks the user to accept the key
func reviewKey() (bool, error) {
	in := bufio.NewReader(os.Stdin)
	for {
		stdout.Printf("Are you sure you want to trust this key (yes/no)?")
		input, err := in.ReadString('\n')
		if err != nil {
			return false, errwrap.Wrap(errors.New("error reading input"), err)
		}
		switch input {
		case "yes\n":
			return true, nil
		case "no\n":
			return false, nil
		default:
			stdout.Printf("Please enter 'yes' or 'no'")
		}
	}
}

func fingerToString(fpr [20]byte) string {
	str := ""
	for i, b := range fpr {
		if i > 0 && i%2 == 0 {
			str += " "
			if i == 10 {
				str += " "
			}
		}
		str += strings.ToUpper(fmt.Sprintf("%.2x", b))
	}
	return str
}
