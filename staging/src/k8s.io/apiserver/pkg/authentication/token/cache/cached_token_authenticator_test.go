/*
Copyright 2017 The Kubernetes Authors.

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

package cache

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	mathrand "math/rand"
	"reflect"
	"sync"
	"testing"
	"time"

	utilclock "k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestCachedTokenAuthenticator(t *testing.T) {
	var (
		calledWithToken []string

		resultUsers map[string]user.Info
		resultOk    bool
		resultErr   error
	)
	fakeAuth := authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		calledWithToken = append(calledWithToken, token)
		return &authenticator.Response{User: resultUsers[token]}, resultOk, resultErr
	})
	fakeClock := utilclock.NewFakeClock(time.Now())

	a := newWithClock(fakeAuth, true, time.Minute, 0, fakeClock)

	calledWithToken, resultUsers, resultOk, resultErr = []string{}, nil, false, nil
	a.AuthenticateToken(context.Background(), "bad1")
	a.AuthenticateToken(context.Background(), "bad2")
	a.AuthenticateToken(context.Background(), "bad3")
	fakeClock.Step(2 * time.Microsecond)
	a.AuthenticateToken(context.Background(), "bad1")
	a.AuthenticateToken(context.Background(), "bad2")
	a.AuthenticateToken(context.Background(), "bad3")
	fakeClock.Step(2 * time.Microsecond)
	if !reflect.DeepEqual(calledWithToken, []string{"bad1", "bad2", "bad3", "bad1", "bad2", "bad3"}) {
		t.Errorf("Expected failing calls to not stay in the cache, got %v", calledWithToken)
	}

	// reset calls, make the backend return success for three user tokens
	calledWithToken = []string{}
	resultUsers, resultOk, resultErr = map[string]user.Info{}, true, nil
	resultUsers["usertoken1"] = &user.DefaultInfo{Name: "user1"}
	resultUsers["usertoken2"] = &user.DefaultInfo{Name: "user2"}
	resultUsers["usertoken3"] = &user.DefaultInfo{Name: "user3"}

	// populate cache
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken1"); err != nil || !ok || resp.User.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken2"); err != nil || !ok || resp.User.GetName() != "user2" {
		t.Errorf("Expected user2")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken3"); err != nil || !ok || resp.User.GetName() != "user3" {
		t.Errorf("Expected user3")
	}
	if !reflect.DeepEqual(calledWithToken, []string{"usertoken1", "usertoken2", "usertoken3"}) {
		t.Errorf("Expected token calls, got %v", calledWithToken)
	}

	// reset calls, make the backend return failures
	calledWithToken = []string{}
	resultUsers, resultOk, resultErr = nil, false, nil

	// authenticate calls still succeed and backend is not hit
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken1"); err != nil || !ok || resp.User.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken2"); err != nil || !ok || resp.User.GetName() != "user2" {
		t.Errorf("Expected user2")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken3"); err != nil || !ok || resp.User.GetName() != "user3" {
		t.Errorf("Expected user3")
	}
	if !reflect.DeepEqual(calledWithToken, []string{}) {
		t.Errorf("Expected no token calls, got %v", calledWithToken)
	}

	// skip forward in time
	fakeClock.Step(2 * time.Minute)

	// backend is consulted again and fails
	a.AuthenticateToken(context.Background(), "usertoken1")
	a.AuthenticateToken(context.Background(), "usertoken2")
	a.AuthenticateToken(context.Background(), "usertoken3")
	if !reflect.DeepEqual(calledWithToken, []string{"usertoken1", "usertoken2", "usertoken3"}) {
		t.Errorf("Expected token calls, got %v", calledWithToken)
	}
}

func TestCachedTokenAuthenticatorWithAudiences(t *testing.T) {
	resultUsers := make(map[string]user.Info)
	fakeAuth := authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		auds, _ := authenticator.AudiencesFrom(ctx)
		return &authenticator.Response{User: resultUsers[auds[0]+token]}, true, nil
	})
	fakeClock := utilclock.NewFakeClock(time.Now())

	a := newWithClock(fakeAuth, true, time.Minute, 0, fakeClock)

	resultUsers["audAusertoken1"] = &user.DefaultInfo{Name: "user1"}
	resultUsers["audBusertoken1"] = &user.DefaultInfo{Name: "user1-different"}

	if u, ok, _ := a.AuthenticateToken(authenticator.WithAudiences(context.Background(), []string{"audA"}), "usertoken1"); !ok || u.User.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if u, ok, _ := a.AuthenticateToken(authenticator.WithAudiences(context.Background(), []string{"audB"}), "usertoken1"); !ok || u.User.GetName() != "user1-different" {
		t.Errorf("Expected user1-different")
	}
}

var bKey string

// use a realistic token for benchmarking
const jwtToken = `eyJhbGciOiJSUzI1NiIsImtpZCI6IiJ9.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJvcGVuc2hpZnQtc2RuIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6InNkbi10b2tlbi1nNndtYyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJzZG4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiIzYzM5YzNhYS1kM2Q5LTExZTktYTVkMC0wMmI3YjllODg1OWUiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6b3BlbnNoaWZ0LXNkbjpzZG4ifQ.PIs0rsUTekj5AX8yJeLDyW4vQB17YS4IOgO026yjEvsCY7Wv_2TD0lwyZWqyQh639q3jPh2_3LTQq2Cp0cReBP1PYOIGgprNm3C-3OFZRnkls-GH09kvPYE8J_-a1YwjxucOwytzJvEM5QTC9iXfEJNSTBfLge-HMYT1y0AGKs8DWTSC4rtd_2PedK3OYiAyDg_xHA8qNpG9pRNM8vfjV9VsmqJtlbnTVlTngqC0t5vyMaWrmLNRxN0rTbN2W9L3diXRnYqI8BUfgPQb7uhYcPuXGeypaFrN4d3yNN4NbgVxnkgdd2IXQ8elSJuQn6ynrvLgG0JPMmThOHnwvsZDeA`

func BenchmarkKeyFunc(b *testing.B) {
	randomCacheKey := make([]byte, 32)
	if _, err := rand.Read(randomCacheKey); err != nil {
		b.Fatal(err) // rand should never fail
	}
	hashPool := &sync.Pool{
		New: func() interface{} {
			return hmac.New(sha256.New, randomCacheKey)
		},
	}

	// use realistic audiences for benchmarking
	auds := []string{"7daf30b7-a85c-429b-8b21-e666aecbb235", "c22aa267-bdde-4acb-8505-998be7818400", "44f9b4f3-7125-4333-b04c-1446a16c6113"}

	b.Run("has audiences", func(b *testing.B) {
		var key string
		for n := 0; n < b.N; n++ {
			key = keyFunc(hashPool, auds, jwtToken)
		}
		bKey = key
	})

	b.Run("nil audiences", func(b *testing.B) {
		var key string
		for n := 0; n < b.N; n++ {
			key = keyFunc(hashPool, nil, jwtToken)
		}
		bKey = key
	})
}

func BenchmarkCachedTokenAuthenticator(b *testing.B) {
	tokenCount := []int{100, 500, 2500, 12500, 62500}
	for _, tc := range tokenCount {
		b.Run(fmt.Sprintf("toks-%v", tc), newSingleBenchmark(tc).bench)
	}
}

func newSingleBenchmark(tokenCount int) *singleBenchmark {
	s := &singleBenchmark{}
	s.makeTokens(tokenCount)
	return s
}

// singleBenchmark collects all the state needed to run a benchmark. The
// question this benchmark answers is, "what's the average latency added by the
// cache for N concurrent tokens?"
type singleBenchmark struct {
	// These token.* variables are set by makeTokens()
	tokenCount int
	// pre-computed response for a token
	tokenToResponse map[string]*cacheRecord
	// include audiences for some
	tokenToAuds map[string]authenticator.Audiences
	// a list makes it easy to select a random one
	tokens []string

	// Simulate slowness, qps limit, external service limitation, etc
	chokepoint chan struct{}

	b  *testing.B
	wg sync.WaitGroup
}

func (s *singleBenchmark) makeTokens(count int) {
	s.tokenCount = count
	s.tokenToResponse = map[string]*cacheRecord{}
	s.tokenToAuds = map[string]authenticator.Audiences{}
	s.tokens = []string{}

	for i := 0; i < s.tokenCount; i++ {
		tok := fmt.Sprintf("%v-%v", jwtToken, i)
		r := cacheRecord{
			resp: &authenticator.Response{
				User: &user.DefaultInfo{Name: fmt.Sprintf("holder of token %v", i)},
			},
		}
		// make different combinations of audience, failures, denies for the tokens.
		auds := []string{}
		for i := 0; i < mathrand.Intn(4); i++ {
			auds = append(auds, string(uuid.NewUUID()))
		}
		choice := mathrand.Intn(1000)
		switch {
		case choice < 900:
			r.ok = true
			r.err = nil
		case choice < 990:
			r.ok = false
			r.err = nil
		default:
			r.ok = false
			r.err = fmt.Errorf("I can't think of a clever error name right now")
		}
		s.tokens = append(s.tokens, tok)
		s.tokenToResponse[tok] = &r
		if len(auds) > 0 {
			s.tokenToAuds[tok] = auds
		}
	}
}

func (s *singleBenchmark) lookup(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	<-s.chokepoint
	defer func() { s.chokepoint <- struct{}{} }()
	time.Sleep(1 * time.Millisecond)
	r, ok := s.tokenToResponse[token]
	if !ok {
		panic("test setup problem")
	}
	return r.resp, r.ok, r.err
}

// To prevent contention over a channel, and to minimize the case where some
// goroutines finish before others, vary the number of goroutines and batch
// size based on the benchmark size.
func (s *singleBenchmark) queueBatches() (<-chan int, int) {
	batchSize := 1
	threads := 1

	switch {
	case s.b.N < 5000:
		threads = s.b.N
		batchSize = 1
	default:
		threads = 5000
		batchSize = s.b.N / (threads * 10)
		if batchSize < 1 {
			batchSize = 1
		}
	}

	batches := make(chan int, threads*2)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer close(batches)
		remaining := s.b.N
		for remaining > batchSize {
			batches <- batchSize
			remaining -= batchSize
		}
		batches <- remaining
	}()

	return batches, threads
}

func (s *singleBenchmark) doAuthForTokenN(n int, a authenticator.Token) {
	tok := s.tokens[n]
	auds := s.tokenToAuds[tok]
	ctx := context.Background()
	ctx = authenticator.WithAudiences(ctx, auds)
	a.AuthenticateToken(ctx, tok)
}

func (s *singleBenchmark) bench(b *testing.B) {
	s.b = b
	a := newWithClock(
		authenticator.TokenFunc(s.lookup),
		true,
		4*time.Second,
		500*time.Millisecond,
		utilclock.RealClock{},
	)
	const maxInFlight = 40
	s.chokepoint = make(chan struct{}, maxInFlight)
	for i := 0; i < maxInFlight; i++ {
		s.chokepoint <- struct{}{}
	}

	batches, threadCount := s.queueBatches()
	s.b.ResetTimer()

	for i := 0; i < threadCount; i++ {
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			// don't contend over the lock for the global rand.Rand
			r := mathrand.New(mathrand.NewSource(mathrand.Int63()))
			for count := range batches {
				for i := 0; i < count; i++ {
					// some problems appear with random
					// access, some appear with many
					// requests for a single entry, so we
					// do both.
					s.doAuthForTokenN(r.Intn(len(s.tokens)), a)
					s.doAuthForTokenN(0, a)
				}
			}
		}()
	}

	s.wg.Wait()
}
