// Copyright 2016 The etcd Authors
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

package main

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"

	clientV2 "github.com/coreos/etcd/client"
)

type v2Stresser struct {
	Endpoint string

	keySize        int
	keySuffixRange int

	N int

	rateLimiter *rate.Limiter

	wg sync.WaitGroup

	atomicModifiedKey int64

	cancel func()
}

func (s *v2Stresser) Stress() error {
	cfg := clientV2.Config{
		Endpoints: []string{s.Endpoint},
		Transport: &http.Transport{
			Dial: (&net.Dialer{
				Timeout:   time.Second,
				KeepAlive: 30 * time.Second,
			}).Dial,
			MaxIdleConnsPerHost: s.N,
		},
	}
	c, err := clientV2.New(cfg)
	if err != nil {
		return err
	}

	kv := clientV2.NewKeysAPI(c)
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel
	s.wg.Add(s.N)
	for i := 0; i < s.N; i++ {
		go func() {
			defer s.wg.Done()
			s.run(ctx, kv)
		}()
	}
	return nil
}

func (s *v2Stresser) run(ctx context.Context, kv clientV2.KeysAPI) {
	for {
		if err := s.rateLimiter.Wait(ctx); err == context.Canceled {
			return
		}
		setctx, setcancel := context.WithTimeout(ctx, clientV2.DefaultRequestTimeout)
		key := fmt.Sprintf("foo%016x", rand.Intn(s.keySuffixRange))
		_, err := kv.Set(setctx, key, string(randBytes(s.keySize)), nil)
		if err == nil {
			atomic.AddInt64(&s.atomicModifiedKey, 1)
		}
		setcancel()
		if err == context.Canceled {
			return
		}
	}
}

func (s *v2Stresser) Cancel() {
	s.cancel()
	s.wg.Wait()
}

func (s *v2Stresser) ModifiedKeys() int64 {
	return atomic.LoadInt64(&s.atomicModifiedKey)
}

func (s *v2Stresser) Checker() Checker { return nil }

func randBytes(size int) []byte {
	data := make([]byte, size)
	for i := 0; i < size; i++ {
		data[i] = byte(int('a') + rand.Intn(26))
	}
	return data
}
