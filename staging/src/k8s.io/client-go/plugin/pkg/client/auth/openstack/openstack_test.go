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

package openstack

import (
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeTokenGetter struct {
	tok *openstackToken
	err error
}

func (f *fakeTokenGetter) Token() (*openstackToken, error) {
	return f.tok, f.err
}

type fakePersister struct {
	lk    sync.Mutex
	cache map[string]string
}

func (f *fakePersister) Persist(cache map[string]string) error {
	f.lk.Lock()
	defer f.lk.Unlock()
	f.cache = map[string]string{}
	for k, v := range cache {
		f.cache[k] = v
	}
	return nil
}

func (f *fakePersister) read() map[string]string {
	ret := map[string]string{}
	f.lk.Lock()
	defer f.lk.Unlock()
	for k, v := range f.cache {
		ret[k] = v
	}
	return ret
}

func TestGettingCachedOrNewToken(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name      string
		cache     map[string]string
		cachedTok *openstackToken
		newTok    *openstackToken
		wantErr   error
		wantTok   *openstackToken
	}{
		{
			"token not in cache, must get new token",
			map[string]string{},
			nil,
			&openstackToken{
				ID:        "validNewToken",
				ExpiresAt: now.Add(time.Hour),
			},
			nil,
			&openstackToken{
				ID:        "validNewToken",
				ExpiresAt: now.Add(time.Hour),
			},
		},
		{
			"cached token is valid, must use cached token",
			map[string]string{
				"token-id":   "validCachedToken",
				"expires-at": now.Add(time.Hour).Format(time.RFC3339Nano),
			},
			&openstackToken{
				ID:        "validCachedToken",
				ExpiresAt: now.Add(time.Hour),
			},
			&openstackToken{
				ID:        "validNewToken",
				ExpiresAt: now.Add(time.Hour),
			},
			nil,
			&openstackToken{
				ID:        "validCachedToken",
				ExpiresAt: now.Add(time.Hour),
			},
		},
		{
			"cached token is expired, must get new token",
			map[string]string{
				"token-id":   "expiredCachedToken",
				"expires-at": now.Add(-time.Hour).Format(time.RFC3339Nano),
			},
			&openstackToken{
				ID:        "expiredCachedToken",
				ExpiresAt: now.Add(-time.Hour),
			},
			&openstackToken{
				ID:        "validNewToken",
				ExpiresAt: now.Add(time.Hour),
			},
			nil,
			&openstackToken{
				ID:        "validNewToken",
				ExpiresAt: now.Add(time.Hour),
			},
		},
		{
			"no token in cache and unable to get new token, must throw error",
			map[string]string{},
			nil,
			nil,
			fmt.Errorf("error getting openstack token"),
			nil,
		},
	}

	for _, tc := range tests {
		persister := &fakePersister{}
		tokenGetter := &fakeTokenGetter{
			tok: tc.newTok,
			err: tc.wantErr,
		}
		cachedGetter := newCachedGetter(tokenGetter, persister, tc.cache)
		gotTok, gotErr := cachedGetter.Token()
		if gotErr != nil {
			if !errEquiv(gotErr, tc.wantErr) {
				t.Errorf("%q Token() error: got %v, want %v", tc.name, gotErr, tc.wantErr)
			}
			continue
		}
		if !(gotTok.ID == tc.wantTok.ID && gotTok.ExpiresAt.Equal(tc.wantTok.ExpiresAt)) {
			t.Errorf("%q Token() got %v, want %v", tc.name, gotTok, tc.wantTok)
		}
	}
}

type MockTransport struct {
	res *http.Response
}

func (t *MockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.res, nil
}

func TestClearingCache(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name      string
		res       http.Response
		cache     map[string]string
		wantCache map[string]string
	}{
		{
			"unauthorized request, must clear cache",
			http.Response{StatusCode: 401},
			map[string]string{
				"token-id":   "unexpiredToken",
				"expires-at": now.Add(time.Hour).Format(time.RFC3339Nano),
			},
			map[string]string{},
		},
		{
			"authorized request, must preserve cache",
			http.Response{StatusCode: 200},
			map[string]string{
				"token-id":   "validToken",
				"expires-at": now.Add(time.Hour).Format(time.RFC3339Nano),
			},
			map[string]string{
				"token-id":   "validToken",
				"expires-at": now.Add(time.Hour).Format(time.RFC3339Nano),
			},
		},
	}

	for _, tc := range tests {
		persister := &fakePersister{}
		tokenGetter := &fakeTokenGetter{}
		cachedGetter := newCachedGetter(tokenGetter, persister, tc.cache)
		oap := &openstackAuthProvider{
			cachedGetter,
			persister,
		}
		persister.Persist(tc.cache)

		req := http.Request{Header: http.Header{}}
		fakeTransport := MockTransport{&tc.res}
		transport := (oap.WrapTransport(&fakeTransport))
		transport.RoundTrip(&req)

		if got := persister.read(); !reflect.DeepEqual(got, tc.wantCache) {
			t.Errorf("%q WrapTransport(): got cache %v, want %v", tc.name, got, tc.wantCache)
		}
	}
}

func TestCacheConcurrentWrites(t *testing.T) {
	now := time.Now()
	tok := &openstackToken{
		ID:        "validNewToken",
		ExpiresAt: now.Add(time.Hour),
	}
	persister := &fakePersister{}
	tokenGetter := &fakeTokenGetter{
		tok,
		nil,
	}
	cache := map[string]string{
		"foo": "bar",
		"baz": "bazinga",
	}
	cachedGetter := newCachedGetter(tokenGetter, persister, cache)

	var wg sync.WaitGroup
	wg.Add(10)
	for i := 0; i < 10; i++ {
		go func() {
			_, err := cachedGetter.Token()
			if err != nil {
				t.Errorf("unexpected error: %s", err)
			}
			wg.Done()
		}()
	}
	wg.Wait()

	cache["token-id"] = tok.ID
	cache["expires-at"] = tok.ExpiresAt.Format(time.RFC3339Nano)
	if got := persister.read(); !reflect.DeepEqual(got, cache) {
		t.Errorf("got cache %v, want %v", got, cache)
	}
}

func errEquiv(got, want error) bool {
	if got == want {
		return true
	}
	if got != nil && want != nil {
		return strings.Contains(got.Error(), want.Error())
	}
	return false
}
