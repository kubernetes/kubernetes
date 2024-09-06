package oidc

import (
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"
	"time"

	"github.com/pquerna/cachecontrol"
	jose "gopkg.in/square/go-jose.v2"
	"k8s.io/klog/v2"
)

// keysExpiryDelta is the allowed clock skew between a client and the OpenID Connect
// server.
//
// When keys expire, they are valid for this amount of time after.
//
// If the keys have not expired, and an ID Token claims it was signed by a key not in
// the cache, if and only if the keys expire in this amount of time, the keys will be
// updated.
const keysExpiryDelta = 30 * time.Second

// NewRemoteKeySet returns a KeySet that can validate JSON web tokens by using HTTP
// GETs to fetch JSON web token sets hosted at a remote URL. This is automatically
// used by NewProvider using the URLs returned by OpenID Connect discovery, but is
// exposed for providers that don't support discovery or to prevent round trips to the
// discovery URL.
//
// The returned KeySet is a long lived verifier that caches keys based on cache-control
// headers. Reuse a common remote key set instead of creating new ones as needed.
//
// The behavior of the returned KeySet is undefined once the context is canceled.
func NewRemoteKeySet(ctx context.Context, jwksURL string) KeySet {
	return newRemoteKeySet(ctx, jwksURL, time.Now)
}

func newRemoteKeySet(ctx context.Context, jwksURL string, now func() time.Time) *remoteKeySet {
	if now == nil {
		now = time.Now
	}
	return &remoteKeySet{jwksURL: jwksURL, ctx: ctx, now: now}
}

type remoteKeySet struct {
	jwksURL string
	ctx     context.Context
	now     func() time.Time

	// guard all other fields
	mu sync.Mutex

	// inflight suppresses parallel execution of updateKeys and allows
	// multiple goroutines to wait for its result.
	inflight *inflight

	// A set of cached keys and their expiry.
	cachedKeys []jose.JSONWebKey
	expiry     time.Time
}

// inflight is used to wait on some in-flight request from multiple goroutines.
type inflight struct {
	doneCh chan struct{}

	keys []jose.JSONWebKey
	err  error
}

func newInflight() *inflight {
	return &inflight{doneCh: make(chan struct{})}
}

// wait returns a channel that multiple goroutines can receive on. Once it returns
// a value, the inflight request is done and result() can be inspected.
func (i *inflight) wait() <-chan struct{} {
	return i.doneCh
}

// done can only be called by a single goroutine. It records the result of the
// inflight request and signals other goroutines that the result is safe to
// inspect.
func (i *inflight) done(keys []jose.JSONWebKey, err error) {
	i.keys = keys
	i.err = err
	close(i.doneCh)
}

// result cannot be called until the wait() channel has returned a value.
func (i *inflight) result() ([]jose.JSONWebKey, error) {
	return i.keys, i.err
}

func (r *remoteKeySet) VerifySignature(ctx context.Context, jwt string) ([]byte, error) {
	jws, err := jose.ParseSigned(jwt)
	if err != nil {
		return nil, fmt.Errorf("oidc: malformed jwt: %v", err)
	}
	return r.verify(ctx, jws)
}

func (r *remoteKeySet) verify(ctx context.Context, jws *jose.JSONWebSignature) ([]byte, error) {
	// We don't support JWTs signed with multiple signatures.
	keyID := ""
	for _, sig := range jws.Signatures {
		keyID = sig.Header.KeyID
		break
	}

	keys, expiry := r.keysFromCache()

	// Don't check expiry yet. This optimizes for when the provider is unavailable.
	kids := make([]string, len(keys))
	for i, k := range keys {
		kids[i] = k.KeyID
	}
	klog.Infof(
		"verifying jwt with kid %s, available kids: %+v, expiry: %s",
		keyID,
		kids,
		expiry.Format(time.RFC3339),
	)
	for _, key := range keys {
		if keyID == "" || key.KeyID == keyID {
			if payload, err := jws.Verify(&key); err == nil {
				return payload, nil
			}
		}
	}

	if !r.now().Add(keysExpiryDelta).After(expiry) {
		// Keys haven't expired, don't refresh.
		return nil, errors.New("failed to verify id token signature")
	}

	klog.Infof("cached JWKS keyset does not contain kid %s, fetching new keyset", keyID)
	keys, err := r.keysFromRemote(ctx)
	if err != nil {
		return nil, fmt.Errorf("fetching keys %v", err)
	}

	kids = make([]string, len(keys))
	for i, key := range keys {
		kids[i] = key.KeyID
		if keyID == "" || key.KeyID == keyID {
			if payload, err := jws.Verify(&key); err == nil {
				return payload, nil
			}
		}
	}
	return nil, fmt.Errorf("failed to verify id token signature for kid %s, available kids %v", keyID, kids)
}

func (r *remoteKeySet) keysFromCache() (keys []jose.JSONWebKey, expiry time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cachedKeys, r.expiry
}

// keysFromRemote syncs the key set from the remote set, records the values in the
// cache, and returns the key set.
func (r *remoteKeySet) keysFromRemote(ctx context.Context) ([]jose.JSONWebKey, error) {
	// Need to lock to inspect the inflight request field.
	r.mu.Lock()
	// If there's not a current inflight request, create one.
	if r.inflight == nil {
		r.inflight = newInflight()

		// This goroutine has exclusive ownership over the current inflight
		// request. It releases the resource by nil'ing the inflight field
		// once the goroutine is done.
		go func() {
			// Sync keys and finish inflight when that's done.
			keys, expiry, err := r.updateKeys()

			r.inflight.done(keys, err)

			// Lock to update the keys and indicate that there is no longer an
			// inflight request.
			r.mu.Lock()
			defer r.mu.Unlock()

			if err == nil {
				r.cachedKeys = keys
				r.expiry = expiry
			}

			// Free inflight so a different request can run.
			r.inflight = nil
		}()
	}
	inflight := r.inflight
	r.mu.Unlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-inflight.wait():
		return inflight.result()
	}
}

func (r *remoteKeySet) updateKeys() ([]jose.JSONWebKey, time.Time, error) {
	req, err := http.NewRequest("GET", r.jwksURL, nil)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("oidc: can't create request: %v", err)
	}

	resp, err := doRequest(r.ctx, req)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("oidc: get keys failed %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("unable to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, time.Time{}, fmt.Errorf("oidc: get keys failed: %s %s", resp.Status, body)
	}

	var keySet jose.JSONWebKeySet
	err = unmarshalResp(resp, body, &keySet)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("oidc: failed to decode keys: %v %s", err, body)
	}
	kids := make([]string, len(keySet.Keys))
	for i, k := range keySet.Keys {
		kids[i] = k.KeyID
	}

	klog.Infof(
		"http response header",
		logHeaders(resp.Header),
		"kids: %+v",
		kids,
		"url: %s",
		r.jwksURL,
	)

	klog.Infof("got %d keys from %s. kids: %+v", len(kids), r.jwksURL, kids)

	// If the server doesn't provide cache control headers, assume the
	// keys expire immediately.
	expiry := r.now()

	_, e, err := cachecontrol.CachableResponse(req, resp, cachecontrol.Options{})
	if err == nil && e.After(expiry) {
		expiry = e
	}
	return keySet.Keys, expiry, nil
}

func logHeaders(r http.Header) string {
	multiline := ""
	for name, values := range r.Clone() {
		for _, value := range values {
			multiline = fmt.Sprintf("%s\n%s: %s", multiline, name, value)
		}
	}
	return multiline
}
