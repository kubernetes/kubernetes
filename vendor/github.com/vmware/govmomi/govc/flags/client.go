/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package flags

import (
	"crypto/sha1"
	"crypto/tls"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
	"golang.org/x/net/context"
)

const (
	envURL           = "GOVC_URL"
	envUsername      = "GOVC_USERNAME"
	envPassword      = "GOVC_PASSWORD"
	envCertificate   = "GOVC_CERTIFICATE"
	envPrivateKey    = "GOVC_PRIVATE_KEY"
	envInsecure      = "GOVC_INSECURE"
	envPersist       = "GOVC_PERSIST_SESSION"
	envMinAPIVersion = "GOVC_MIN_API_VERSION"
	envVimNamespace  = "GOVC_VIM_NAMESPACE"
	envVimVersion    = "GOVC_VIM_VERSION"
)

const cDescr = "ESX or vCenter URL"

type ClientFlag struct {
	common

	*DebugFlag

	url           *url.URL
	username      string
	password      string
	cert          string
	key           string
	insecure      bool
	persist       bool
	minAPIVersion string
	vimNamespace  string
	vimVersion    string

	client *vim25.Client
}

var clientFlagKey = flagKey("client")

func NewClientFlag(ctx context.Context) (*ClientFlag, context.Context) {
	if v := ctx.Value(clientFlagKey); v != nil {
		return v.(*ClientFlag), ctx
	}

	v := &ClientFlag{}
	v.DebugFlag, ctx = NewDebugFlag(ctx)
	ctx = context.WithValue(ctx, clientFlagKey, v)
	return v, ctx
}

func (flag *ClientFlag) URLWithoutPassword() *url.URL {
	if flag.url == nil {
		return nil
	}

	withoutCredentials := *flag.url
	withoutCredentials.User = url.User(flag.url.User.Username())
	return &withoutCredentials
}

func (flag *ClientFlag) String() string {
	url := flag.URLWithoutPassword()
	if url == nil {
		return ""
	}

	return url.String()
}

func (flag *ClientFlag) Set(s string) error {
	var err error

	flag.url, err = soap.ParseURL(s)

	return err
}

func (flag *ClientFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.RegisterOnce(func() {
		flag.DebugFlag.Register(ctx, f)

		{
			flag.Set(os.Getenv(envURL))
			usage := fmt.Sprintf("%s [%s]", cDescr, envURL)
			f.Var(flag, "u", usage)
		}

		{
			flag.username = os.Getenv(envUsername)
			flag.password = os.Getenv(envPassword)
		}

		{
			value := os.Getenv(envCertificate)
			usage := fmt.Sprintf("Certificate [%s]", envCertificate)
			f.StringVar(&flag.cert, "cert", value, usage)
		}

		{
			value := os.Getenv(envPrivateKey)
			usage := fmt.Sprintf("Private key [%s]", envPrivateKey)
			f.StringVar(&flag.key, "key", value, usage)
		}

		{
			insecure := false
			switch env := strings.ToLower(os.Getenv(envInsecure)); env {
			case "1", "true":
				insecure = true
			}

			usage := fmt.Sprintf("Skip verification of server certificate [%s]", envInsecure)
			f.BoolVar(&flag.insecure, "k", insecure, usage)
		}

		{
			persist := true
			switch env := strings.ToLower(os.Getenv(envPersist)); env {
			case "0", "false":
				persist = false
			}

			usage := fmt.Sprintf("Persist session to disk [%s]", envPersist)
			f.BoolVar(&flag.persist, "persist-session", persist, usage)
		}

		{
			env := os.Getenv(envMinAPIVersion)
			if env == "" {
				env = "5.5"
			}

			flag.minAPIVersion = env
		}

		{
			value := os.Getenv(envVimNamespace)
			if value == "" {
				value = soap.DefaultVimNamespace
			}
			usage := fmt.Sprintf("Vim namespace [%s]", envVimNamespace)
			f.StringVar(&flag.vimNamespace, "vim-namespace", value, usage)
		}

		{
			value := os.Getenv(envVimVersion)
			if value == "" {
				value = soap.DefaultVimVersion
			}
			usage := fmt.Sprintf("Vim version [%s]", envVimVersion)
			f.StringVar(&flag.vimVersion, "vim-version", value, usage)
		}
	})
}

func (flag *ClientFlag) Process(ctx context.Context) error {
	return flag.ProcessOnce(func() error {
		if err := flag.DebugFlag.Process(ctx); err != nil {
			return err
		}

		if flag.url == nil {
			return errors.New("specify an " + cDescr)
		}

		// Override username if set
		if flag.username != "" {
			var password string
			var ok bool

			if flag.url.User != nil {
				password, ok = flag.url.User.Password()
			}

			if ok {
				flag.url.User = url.UserPassword(flag.username, password)
			} else {
				flag.url.User = url.User(flag.username)
			}
		}

		// Override password if set
		if flag.password != "" {
			var username string

			if flag.url.User != nil {
				username = flag.url.User.Username()
			}

			flag.url.User = url.UserPassword(username, flag.password)
		}

		return nil
	})
}

// Retry twice when a temporary I/O error occurs.
// This means a maximum of 3 attempts.
func attachRetries(rt soap.RoundTripper) soap.RoundTripper {
	return vim25.Retry(rt, vim25.TemporaryNetworkError(3))
}

func (flag *ClientFlag) sessionFile() string {
	url := flag.URLWithoutPassword()

	// Key session file off of full URI and insecure setting.
	// Hash key to get a predictable, canonical format.
	key := fmt.Sprintf("%s#insecure=%t", url.String(), flag.insecure)
	name := fmt.Sprintf("%040x", sha1.Sum([]byte(key)))
	return filepath.Join(os.Getenv("HOME"), ".govmomi", "sessions", name)
}

func (flag *ClientFlag) saveClient(c *vim25.Client) error {
	if !flag.persist {
		return nil
	}

	p := flag.sessionFile()
	err := os.MkdirAll(filepath.Dir(p), 0700)
	if err != nil {
		return err
	}

	f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()

	err = json.NewEncoder(f).Encode(c)
	if err != nil {
		return err
	}

	return nil
}

func (flag *ClientFlag) restoreClient(c *vim25.Client) (bool, error) {
	if !flag.persist {
		return false, nil
	}

	f, err := os.Open(flag.sessionFile())
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}

		return false, err
	}

	defer f.Close()

	dec := json.NewDecoder(f)
	err = dec.Decode(c)
	if err != nil {
		return false, err
	}

	return true, nil
}

func (flag *ClientFlag) loadClient() (*vim25.Client, error) {
	c := new(vim25.Client)
	ok, err := flag.restoreClient(c)
	if err != nil {
		return nil, err
	}

	if !ok || !c.Valid() {
		return nil, nil
	}

	// Add retry functionality before making any calls
	c.RoundTripper = attachRetries(c.RoundTripper)

	m := session.NewManager(c)
	u, err := m.UserSession(context.TODO())
	if err != nil {
		return nil, err
	}

	// If the session is nil, the client is not authenticated
	if u == nil {
		return nil, nil
	}

	return c, nil
}

func (flag *ClientFlag) newClient() (*vim25.Client, error) {
	sc := soap.NewClient(flag.url, flag.insecure)
	isTunnel := false

	if flag.cert != "" {
		isTunnel = true
		cert, err := tls.LoadX509KeyPair(flag.cert, flag.key)
		if err != nil {
			return nil, err
		}

		sc.SetCertificate(cert)
	}

	// Set namespace and version
	sc.Namespace = flag.vimNamespace
	sc.Version = flag.vimVersion

	// Add retry functionality before making any calls
	rt := attachRetries(sc)
	c, err := vim25.NewClient(context.TODO(), rt)
	if err != nil {
		return nil, err
	}

	// Set client, since we didn't pass it in the constructor
	c.Client = sc

	m := session.NewManager(c)
	u := flag.url.User
	if isTunnel {
		err = m.LoginExtensionByCertificate(context.TODO(), u.Username(), "")
		if err != nil {
			return nil, err
		}
	} else {
		err = m.Login(context.TODO(), u)
		if err != nil {
			return nil, err
		}
	}

	err = flag.saveClient(c)
	if err != nil {
		return nil, err
	}

	return c, nil
}

// apiVersionValid returns whether or not the API version supported by the
// server the client is connected to is not recent enough.
func apiVersionValid(c *vim25.Client, minVersionString string) error {
	if minVersionString == "-" {
		// Disable version check
		return nil
	}

	apiVersion := c.ServiceContent.About.ApiVersion
	if strings.HasSuffix(apiVersion, ".x") {
		// Skip version check for development builds
		return nil
	}

	realVersion, err := ParseVersion(apiVersion)
	if err != nil {
		return err
	}

	minVersion, err := ParseVersion(minVersionString)
	if err != nil {
		return err
	}

	if !minVersion.Lte(realVersion) {
		err = fmt.Errorf("Require API version %s, connected to API version %s (set %s to override)",
			minVersionString,
			c.ServiceContent.About.ApiVersion,
			envMinAPIVersion)
		return err
	}

	return nil
}

func (flag *ClientFlag) Client() (*vim25.Client, error) {
	if flag.client != nil {
		return flag.client, nil
	}

	c, err := flag.loadClient()
	if err != nil {
		return nil, err
	}

	// loadClient returns nil if it was unable to load a session from disk
	if c == nil {
		c, err = flag.newClient()
		if err != nil {
			return nil, err
		}
	}

	// Check that the endpoint has the right API version
	err = apiVersionValid(c, flag.minAPIVersion)
	if err != nil {
		return nil, err
	}

	flag.client = c
	return flag.client, nil
}
