/*
Copyright 2018 The Kubernetes Authors.

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

package exec

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"reflect"
	"sync"
	"time"

	"github.com/davecgh/go-spew/spew"
	"golang.org/x/crypto/ssh/terminal"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/pkg/apis/clientauthentication"
	"k8s.io/client-go/pkg/apis/clientauthentication/v1alpha1"
	"k8s.io/client-go/pkg/apis/clientauthentication/v1beta1"
	"k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/metrics"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/connrotation"
	"k8s.io/klog"
)

const execInfoEnv = "KUBERNETES_EXEC_INFO"
const onRotateListWarningLength = 1000

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

func init() {
	v1.AddToGroupVersion(scheme, schema.GroupVersion{Version: "v1"})
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	utilruntime.Must(v1beta1.AddToScheme(scheme))
	utilruntime.Must(clientauthentication.AddToScheme(scheme))
}

var (
	// Since transports can be constantly re-initialized by programs like kubectl,
	// keep a cache of initialized authenticators keyed by a hash of their config.
	globalCache = newCache()
	// The list of API versions we accept.
	apiVersions = map[string]schema.GroupVersion{
		v1alpha1.SchemeGroupVersion.String(): v1alpha1.SchemeGroupVersion,
		v1beta1.SchemeGroupVersion.String():  v1beta1.SchemeGroupVersion,
	}
)

func newCache() *cache {
	return &cache{m: make(map[string]*Authenticator)}
}

var spewConfig = &spew.ConfigState{DisableMethods: true, Indent: " "}

func cacheKey(c *api.ExecConfig) string {
	return spewConfig.Sprint(c)
}

type cache struct {
	mu sync.Mutex
	m  map[string]*Authenticator
}

func (c *cache) get(s string) (*Authenticator, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	a, ok := c.m[s]
	return a, ok
}

// put inserts an authenticator into the cache. If an authenticator is already
// associated with the key, the first one is returned instead.
func (c *cache) put(s string, a *Authenticator) *Authenticator {
	c.mu.Lock()
	defer c.mu.Unlock()
	existing, ok := c.m[s]
	if ok {
		return existing
	}
	c.m[s] = a
	return a
}

// GetAuthenticator returns an exec-based plugin for providing client credentials.
func GetAuthenticator(config *api.ExecConfig) (*Authenticator, error) {
	return newAuthenticator(globalCache, config)
}

func newAuthenticator(c *cache, config *api.ExecConfig) (*Authenticator, error) {
	key := cacheKey(config)
	if a, ok := c.get(key); ok {
		return a, nil
	}

	gv, ok := apiVersions[config.APIVersion]
	if !ok {
		return nil, fmt.Errorf("exec plugin: invalid apiVersion %q", config.APIVersion)
	}

	a := &Authenticator{
		cmd:   config.Command,
		args:  config.Args,
		group: gv,

		stdin:       os.Stdin,
		stderr:      os.Stderr,
		interactive: terminal.IsTerminal(int(os.Stdout.Fd())),
		now:         time.Now,
		environ:     os.Environ,
	}

	for _, env := range config.Env {
		a.env = append(a.env, env.Name+"="+env.Value)
	}

	return c.put(key, a), nil
}

// Authenticator is a client credential provider that rotates credentials by executing a plugin.
// The plugin input and output are defined by the API group client.authentication.k8s.io.
type Authenticator struct {
	// Set by the config
	cmd   string
	args  []string
	group schema.GroupVersion
	env   []string

	// Stubbable for testing
	stdin       io.Reader
	stderr      io.Writer
	interactive bool
	now         func() time.Time
	environ     func() []string

	// Cached results.
	//
	// The mutex also guards calling the plugin. Since the plugin could be
	// interactive we want to make sure it's only called once.
	mu          sync.Mutex
	cachedCreds *credentials
	exp         time.Time

	onRotateList []func()
}

type credentials struct {
	token string
	cert  *tls.Certificate
}

// UpdateTransportConfig updates the transport.Config to use credentials
// returned by the plugin.
func (a *Authenticator) UpdateTransportConfig(c *transport.Config) error {
	c.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return &roundTripper{a, rt}
	})

	if c.TLS.GetCert != nil {
		return errors.New("can't add TLS certificate callback: transport.Config.TLS.GetCert already set")
	}
	c.TLS.GetCert = a.cert

	var dial func(ctx context.Context, network, addr string) (net.Conn, error)
	if c.Dial != nil {
		dial = c.Dial
	} else {
		dial = (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	}
	d := connrotation.NewDialer(dial)

	a.mu.Lock()
	defer a.mu.Unlock()
	a.onRotateList = append(a.onRotateList, d.CloseAll)
	onRotateListLength := len(a.onRotateList)
	if onRotateListLength > onRotateListWarningLength {
		klog.Warningf("constructing many client instances from the same exec auth config can cause performance problems during cert rotation and can exhaust available network connections; %d clients constructed calling %q", onRotateListLength, a.cmd)
	}

	c.Dial = d.DialContext

	return nil
}

type roundTripper struct {
	a    *Authenticator
	base http.RoundTripper
}

func (r *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// If a user has already set credentials, use that. This makes commands like
	// "kubectl get --token (token) pods" work.
	if req.Header.Get("Authorization") != "" {
		return r.base.RoundTrip(req)
	}

	creds, err := r.a.getCreds()
	if err != nil {
		return nil, fmt.Errorf("getting credentials: %v", err)
	}
	if creds.token != "" {
		req.Header.Set("Authorization", "Bearer "+creds.token)
	}

	res, err := r.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode == http.StatusUnauthorized {
		resp := &clientauthentication.Response{
			Header: res.Header,
			Code:   int32(res.StatusCode),
		}
		if err := r.a.maybeRefreshCreds(creds, resp); err != nil {
			klog.Errorf("refreshing credentials: %v", err)
		}
	}
	return res, nil
}

func (a *Authenticator) credsExpired() bool {
	if a.exp.IsZero() {
		return false
	}
	return a.now().After(a.exp)
}

func (a *Authenticator) cert() (*tls.Certificate, error) {
	creds, err := a.getCreds()
	if err != nil {
		return nil, err
	}
	return creds.cert, nil
}

func (a *Authenticator) getCreds() (*credentials, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.cachedCreds != nil && !a.credsExpired() {
		return a.cachedCreds, nil
	}

	if err := a.refreshCredsLocked(nil); err != nil {
		return nil, err
	}

	return a.cachedCreds, nil
}

// maybeRefreshCreds executes the plugin to force a rotation of the
// credentials, unless they were rotated already.
func (a *Authenticator) maybeRefreshCreds(creds *credentials, r *clientauthentication.Response) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Since we're not making a new pointer to a.cachedCreds in getCreds, no
	// need to do deep comparison.
	if creds != a.cachedCreds {
		// Credentials already rotated.
		return nil
	}

	return a.refreshCredsLocked(r)
}

// refreshCredsLocked executes the plugin and reads the credentials from
// stdout. It must be called while holding the Authenticator's mutex.
func (a *Authenticator) refreshCredsLocked(r *clientauthentication.Response) error {
	cred := &clientauthentication.ExecCredential{
		Spec: clientauthentication.ExecCredentialSpec{
			Response:    r,
			Interactive: a.interactive,
		},
	}

	env := append(a.environ(), a.env...)
	if a.group == v1alpha1.SchemeGroupVersion {
		// Input spec disabled for beta due to lack of use. Possibly re-enable this later if
		// someone wants it back.
		//
		// See: https://github.com/kubernetes/kubernetes/issues/61796
		data, err := runtime.Encode(codecs.LegacyCodec(a.group), cred)
		if err != nil {
			return fmt.Errorf("encode ExecCredentials: %v", err)
		}
		env = append(env, fmt.Sprintf("%s=%s", execInfoEnv, data))
	}

	stdout := &bytes.Buffer{}
	cmd := exec.Command(a.cmd, a.args...)
	cmd.Env = env
	cmd.Stderr = a.stderr
	cmd.Stdout = stdout
	if a.interactive {
		cmd.Stdin = a.stdin
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("exec: %v", err)
	}

	_, gvk, err := codecs.UniversalDecoder(a.group).Decode(stdout.Bytes(), nil, cred)
	if err != nil {
		return fmt.Errorf("decoding stdout: %v", err)
	}
	if gvk.Group != a.group.Group || gvk.Version != a.group.Version {
		return fmt.Errorf("exec plugin is configured to use API version %s, plugin returned version %s",
			a.group, schema.GroupVersion{Group: gvk.Group, Version: gvk.Version})
	}

	if cred.Status == nil {
		return fmt.Errorf("exec plugin didn't return a status field")
	}
	if cred.Status.Token == "" && cred.Status.ClientCertificateData == "" && cred.Status.ClientKeyData == "" {
		return fmt.Errorf("exec plugin didn't return a token or cert/key pair")
	}
	if (cred.Status.ClientCertificateData == "") != (cred.Status.ClientKeyData == "") {
		return fmt.Errorf("exec plugin returned only certificate or key, not both")
	}

	if cred.Status.ExpirationTimestamp != nil {
		a.exp = cred.Status.ExpirationTimestamp.Time
	} else {
		a.exp = time.Time{}
	}

	newCreds := &credentials{
		token: cred.Status.Token,
	}
	if cred.Status.ClientKeyData != "" && cred.Status.ClientCertificateData != "" {
		cert, err := tls.X509KeyPair([]byte(cred.Status.ClientCertificateData), []byte(cred.Status.ClientKeyData))
		if err != nil {
			return fmt.Errorf("failed parsing client key/certificate: %v", err)
		}

		// Leaf is initialized to be nil:
		//  https://golang.org/pkg/crypto/tls/#X509KeyPair
		// Leaf certificate is the first certificate:
		//  https://golang.org/pkg/crypto/tls/#Certificate
		// Populating leaf is useful for quickly accessing the underlying x509
		// certificate values.
		cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			return fmt.Errorf("failed parsing client leaf certificate: %v", err)
		}
		newCreds.cert = &cert
	}

	oldCreds := a.cachedCreds
	a.cachedCreds = newCreds
	// Only close all connections when TLS cert rotates. Token rotation doesn't
	// need the extra noise.
	if oldCreds != nil && !reflect.DeepEqual(oldCreds.cert, a.cachedCreds.cert) {
		// Can be nil if the exec auth plugin only returned token auth.
		if oldCreds.cert != nil && oldCreds.cert.Leaf != nil {
			metrics.ClientCertRotationAge.Observe(time.Now().Sub(oldCreds.cert.Leaf.NotBefore))
		}
		for _, onRotate := range a.onRotateList {
			onRotate()
		}
	}

	expiry := time.Time{}
	if a.cachedCreds.cert != nil && a.cachedCreds.cert.Leaf != nil {
		expiry = a.cachedCreds.cert.Leaf.NotAfter
	}
	expirationMetrics.set(a, expiry)
	return nil
}
