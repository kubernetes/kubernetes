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
	"fmt"
	"math/rand"
	"reflect"
	"sync"
	"testing"
	"time"

	utilclock "k8s.io/apimachinery/pkg/util/clock"
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

func BenchmarkCachedTokenAuthenticator(b *testing.B) {
	tokenCount := []int{100, 500, 1000, 2000, 5000, 10000}
	for _, tc := range tokenCount {
		b.Run(fmt.Sprintf("toks-%v", tc), newSingleBenchmark(tc).bench)
	}
}

func newSingleBenchmark(tokenCount int) *singleBenchmark {
	s := &singleBenchmark{
		tokenCount: tokenCount,
	}
	s.makeTokens()
	return s
}

type singleBenchmark struct {
	tokenCount int

	tokenToResponse map[string]*cacheRecord
	tokens          []string
	chokepoint      chan struct{}

	b  *testing.B
	wg sync.WaitGroup
}

func (s *singleBenchmark) makeTokens() {
	s.tokenToResponse = map[string]*cacheRecord{}
	s.tokens = []string{}

	for i := 0; i < s.tokenCount; i++ {
		tok := fmt.Sprintf("token_%v", i)
		r := cacheRecord{
			resp: &authenticator.Response{
				User: &user.DefaultInfo{Name: "holder of token " + tok},
			},
		}
		choice := rand.Intn(1000)
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

	batches := make(chan int, 1000000)
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

func (s *singleBenchmark) bench(b *testing.B) {
	s.b = b
	a := newWithClock(
		authenticator.TokenFunc(s.lookup),
		true,
		4*time.Second,
		500*time.Millisecond,
		utilclock.RealClock{},
	)
	const maxInFlight = 4
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
			r := rand.New(rand.NewSource(rand.Int63()))
			for count := range batches {
				for i := 0; i < count; i++ {
					tok := s.tokens[r.Intn(len(s.tokens))]
					a.AuthenticateToken(context.Background(), tok)
					a.AuthenticateToken(context.Background(), s.tokens[0])
				}
			}
		}()
	}

	s.wg.Wait()
}
