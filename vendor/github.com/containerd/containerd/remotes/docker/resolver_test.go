package docker

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/containerd/containerd/remotes"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

func TestHTTPResolver(t *testing.T) {
	s := func(h http.Handler) (string, ResolverOptions, func()) {
		s := httptest.NewServer(h)

		options := ResolverOptions{
			PlainHTTP: true,
		}
		base := s.URL[7:] // strip "http://"
		return base, options, s.Close
	}

	runBasicTest(t, "testname", s)
}

func TestHTTPSResolver(t *testing.T) {
	runBasicTest(t, "testname", tlsServer)
}

func TestBasicResolver(t *testing.T) {
	basicAuth := func(h http.Handler) (string, ResolverOptions, func()) {
		// Wrap with basic auth
		wrapped := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
			username, password, ok := r.BasicAuth()
			if !ok || username != "user1" || password != "password1" {
				rw.Header().Set("WWW-Authenticate", "Basic realm=localhost")
				rw.WriteHeader(http.StatusUnauthorized)
				return
			}
			h.ServeHTTP(rw, r)
		})

		base, options, close := tlsServer(wrapped)
		options.Credentials = func(string) (string, string, error) {
			return "user1", "password1", nil
		}
		return base, options, close
	}
	runBasicTest(t, "testname", basicAuth)
}

func TestAnonymousTokenResolver(t *testing.T) {
	th := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(http.StatusOK)
		rw.Write([]byte(`{"access_token":"perfectlyvalidopaquetoken"}`))
	})

	runBasicTest(t, "testname", withTokenServer(th, nil))
}

func TestBasicAuthTokenResolver(t *testing.T) {
	th := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(http.StatusOK)
		username, password, ok := r.BasicAuth()
		if !ok || username != "user1" || password != "password1" {
			rw.Write([]byte(`{"access_token":"insufficientscope"}`))
		} else {
			rw.Write([]byte(`{"access_token":"perfectlyvalidopaquetoken"}`))
		}
	})
	creds := func(string) (string, string, error) {
		return "user1", "password1", nil
	}

	runBasicTest(t, "testname", withTokenServer(th, creds))
}

func TestRefreshTokenResolver(t *testing.T) {
	th := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(http.StatusOK)

		r.ParseForm()
		if r.PostForm.Get("grant_type") != "refresh_token" || r.PostForm.Get("refresh_token") != "somerefreshtoken" {
			rw.Write([]byte(`{"access_token":"insufficientscope"}`))
		} else {
			rw.Write([]byte(`{"access_token":"perfectlyvalidopaquetoken"}`))
		}
	})
	creds := func(string) (string, string, error) {
		return "", "somerefreshtoken", nil
	}

	runBasicTest(t, "testname", withTokenServer(th, creds))
}

func TestPostBasicAuthTokenResolver(t *testing.T) {
	th := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(http.StatusOK)

		r.ParseForm()
		if r.PostForm.Get("grant_type") != "password" || r.PostForm.Get("username") != "user1" || r.PostForm.Get("password") != "password1" {
			rw.Write([]byte(`{"access_token":"insufficientscope"}`))
		} else {
			rw.Write([]byte(`{"access_token":"perfectlyvalidopaquetoken"}`))
		}
	})
	creds := func(string) (string, string, error) {
		return "user1", "password1", nil
	}

	runBasicTest(t, "testname", withTokenServer(th, creds))
}

func TestBadTokenResolver(t *testing.T) {
	th := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(http.StatusOK)
		rw.Write([]byte(`{"access_token":"insufficientscope"}`))
	})
	creds := func(string) (string, string, error) {
		return "", "somerefreshtoken", nil
	}

	ctx := context.Background()
	h := newContent(ocispec.MediaTypeImageManifest, []byte("not anything parse-able"))

	base, ro, close := withTokenServer(th, creds)(logHandler{t, h})
	defer close()

	resolver := NewResolver(ro)
	image := fmt.Sprintf("%s/doesntmatter:sometatg", base)

	_, _, err := resolver.Resolve(ctx, image)
	if err == nil {
		t.Fatal("Expected error getting token with inssufficient scope")
	}
	if errors.Cause(err) != ErrInvalidAuthorization {
		t.Fatal(err)
	}
}

func withTokenServer(th http.Handler, creds func(string) (string, string, error)) func(h http.Handler) (string, ResolverOptions, func()) {
	return func(h http.Handler) (string, ResolverOptions, func()) {
		s := httptest.NewUnstartedServer(th)
		s.StartTLS()

		cert, _ := x509.ParseCertificate(s.TLS.Certificates[0].Certificate[0])
		tokenBase := s.URL + "/token"

		// Wrap with token auth
		wrapped := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
			auth := strings.ToLower(r.Header.Get("Authorization"))
			if auth != "bearer perfectlyvalidopaquetoken" {
				authHeader := fmt.Sprintf("Bearer realm=%q,service=registry,scope=\"repository:testname:pull,pull\"", tokenBase)
				if strings.HasPrefix(auth, "bearer ") {
					authHeader = authHeader + ",error=" + auth[7:]
				}
				rw.Header().Set("WWW-Authenticate", authHeader)
				rw.WriteHeader(http.StatusUnauthorized)
				return
			}
			h.ServeHTTP(rw, r)
		})

		base, options, close := tlsServer(wrapped)
		options.Credentials = creds
		options.Client.Transport.(*http.Transport).TLSClientConfig.RootCAs.AddCert(cert)
		return base, options, func() {
			s.Close()
			close()
		}
	}
}

func tlsServer(h http.Handler) (string, ResolverOptions, func()) {
	s := httptest.NewUnstartedServer(h)
	s.StartTLS()

	capool := x509.NewCertPool()
	cert, _ := x509.ParseCertificate(s.TLS.Certificates[0].Certificate[0])
	capool.AddCert(cert)

	options := ResolverOptions{
		Client: &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					RootCAs: capool,
				},
			},
		},
	}
	base := s.URL[8:] // strip "https://"
	return base, options, s.Close
}

type logHandler struct {
	t       *testing.T
	handler http.Handler
}

func (h logHandler) ServeHTTP(rw http.ResponseWriter, r *http.Request) {
	h.t.Logf("%s %s", r.Method, r.URL.String())
	h.handler.ServeHTTP(rw, r)
}

func runBasicTest(t *testing.T, name string, sf func(h http.Handler) (string, ResolverOptions, func())) {
	var (
		ctx = context.Background()
		tag = "latest"
		r   = http.NewServeMux()
	)

	m := newManifest(
		newContent(ocispec.MediaTypeImageConfig, []byte("1")),
		newContent(ocispec.MediaTypeImageLayerGzip, []byte("2")),
	)
	mc := newContent(ocispec.MediaTypeImageManifest, m.OCIManifest())
	m.RegisterHandler(r, name)
	r.Handle(fmt.Sprintf("/v2/%s/manifests/%s", name, tag), mc)
	r.Handle(fmt.Sprintf("/v2/%s/manifests/%s", name, mc.Digest()), mc)

	base, ro, close := sf(logHandler{t, r})
	defer close()

	resolver := NewResolver(ro)
	image := fmt.Sprintf("%s/%s:%s", base, name, tag)

	_, d, err := resolver.Resolve(ctx, image)
	if err != nil {
		t.Fatal(err)
	}
	f, err := resolver.Fetcher(ctx, image)
	if err != nil {
		t.Fatal(err)
	}

	refs, err := testocimanifest(ctx, f, d)
	if err != nil {
		t.Fatal(err)
	}

	if len(refs) != 2 {
		t.Fatalf("Unexpected number of references: %d, expected 2", len(refs))
	}

	for _, ref := range refs {
		if err := testFetch(ctx, f, ref); err != nil {
			t.Fatal(err)
		}
	}
}

func testFetch(ctx context.Context, f remotes.Fetcher, desc ocispec.Descriptor) error {
	r, err := f.Fetch(ctx, desc)
	if err != nil {
		return err
	}
	dgstr := desc.Digest.Algorithm().Digester()
	io.Copy(dgstr.Hash(), r)
	if dgstr.Digest() != desc.Digest {
		return errors.Errorf("content mismatch: %s != %s", dgstr.Digest(), desc.Digest)
	}

	return nil
}

func testocimanifest(ctx context.Context, f remotes.Fetcher, desc ocispec.Descriptor) ([]ocispec.Descriptor, error) {
	r, err := f.Fetch(ctx, desc)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to fetch %s", desc.Digest)
	}
	p, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if dgst := desc.Digest.Algorithm().FromBytes(p); dgst != desc.Digest {
		return nil, errors.Errorf("digest mismatch: %s != %s", dgst, desc.Digest)
	}

	var manifest ocispec.Manifest
	if err := json.Unmarshal(p, &manifest); err != nil {
		return nil, err
	}

	var descs []ocispec.Descriptor

	descs = append(descs, manifest.Config)
	descs = append(descs, manifest.Layers...)

	return descs, nil
}

type testContent struct {
	mediaType string
	content   []byte
}

func newContent(mediaType string, b []byte) testContent {
	return testContent{
		mediaType: mediaType,
		content:   b,
	}
}

func (tc testContent) Descriptor() ocispec.Descriptor {
	return ocispec.Descriptor{
		MediaType: tc.mediaType,
		Digest:    digest.FromBytes(tc.content),
		Size:      int64(len(tc.content)),
	}
}

func (tc testContent) Digest() digest.Digest {
	return digest.FromBytes(tc.content)
}

func (tc testContent) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", tc.mediaType)
	w.Header().Add("Content-Length", strconv.Itoa(len(tc.content)))
	w.Header().Add("Docker-Content-Digest", tc.Digest().String())
	w.WriteHeader(http.StatusOK)
	w.Write(tc.content)
}

type testManifest struct {
	config     testContent
	references []testContent
}

func newManifest(config testContent, refs ...testContent) testManifest {
	return testManifest{
		config:     config,
		references: refs,
	}
}

func (m testManifest) OCIManifest() []byte {
	manifest := ocispec.Manifest{
		Versioned: specs.Versioned{
			SchemaVersion: 1,
		},
		Config: m.config.Descriptor(),
		Layers: make([]ocispec.Descriptor, len(m.references)),
	}
	for i, c := range append(m.references) {
		manifest.Layers[i] = c.Descriptor()
	}
	b, _ := json.Marshal(manifest)
	return b
}

func (m testManifest) RegisterHandler(r *http.ServeMux, name string) {
	for _, c := range append(m.references, m.config) {
		r.Handle(fmt.Sprintf("/v2/%s/blobs/%s", name, c.Digest()), c)
	}
}
