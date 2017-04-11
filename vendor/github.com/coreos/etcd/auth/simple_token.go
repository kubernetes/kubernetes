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

package auth

// CAUTION: This randum number based token mechanism is only for testing purpose.
// JWT based mechanism will be added in the near future.

import (
	"crypto/rand"
	"math/big"
	"strings"
	"sync"
	"time"
)

const (
	letters                  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	defaultSimpleTokenLength = 16
)

// var for testing purposes
var (
	simpleTokenTTL           = 5 * time.Minute
	simpleTokenTTLResolution = 1 * time.Second
)

type simpleTokenTTLKeeper struct {
	tokensMu        sync.Mutex
	tokens          map[string]time.Time
	stopCh          chan chan struct{}
	deleteTokenFunc func(string)
}

func NewSimpleTokenTTLKeeper(deletefunc func(string)) *simpleTokenTTLKeeper {
	stk := &simpleTokenTTLKeeper{
		tokens:          make(map[string]time.Time),
		stopCh:          make(chan chan struct{}),
		deleteTokenFunc: deletefunc,
	}
	go stk.run()
	return stk
}

func (tm *simpleTokenTTLKeeper) stop() {
	waitCh := make(chan struct{})
	tm.stopCh <- waitCh
	<-waitCh
	close(tm.stopCh)
}

func (tm *simpleTokenTTLKeeper) addSimpleToken(token string) {
	tm.tokens[token] = time.Now().Add(simpleTokenTTL)
}

func (tm *simpleTokenTTLKeeper) resetSimpleToken(token string) {
	if _, ok := tm.tokens[token]; ok {
		tm.tokens[token] = time.Now().Add(simpleTokenTTL)
	}
}

func (tm *simpleTokenTTLKeeper) deleteSimpleToken(token string) {
	delete(tm.tokens, token)
}

func (tm *simpleTokenTTLKeeper) run() {
	tokenTicker := time.NewTicker(simpleTokenTTLResolution)
	defer tokenTicker.Stop()
	for {
		select {
		case <-tokenTicker.C:
			nowtime := time.Now()
			tm.tokensMu.Lock()
			for t, tokenendtime := range tm.tokens {
				if nowtime.After(tokenendtime) {
					tm.deleteTokenFunc(t)
					delete(tm.tokens, t)
				}
			}
			tm.tokensMu.Unlock()
		case waitCh := <-tm.stopCh:
			tm.tokens = make(map[string]time.Time)
			waitCh <- struct{}{}
			return
		}
	}
}

func (as *authStore) GenSimpleToken() (string, error) {
	ret := make([]byte, defaultSimpleTokenLength)

	for i := 0; i < defaultSimpleTokenLength; i++ {
		bInt, err := rand.Int(rand.Reader, big.NewInt(int64(len(letters))))
		if err != nil {
			return "", err
		}

		ret[i] = letters[bInt.Int64()]
	}

	return string(ret), nil
}

func (as *authStore) assignSimpleTokenToUser(username, token string) {
	as.simpleTokenKeeper.tokensMu.Lock()
	as.simpleTokensMu.Lock()

	_, ok := as.simpleTokens[token]
	if ok {
		plog.Panicf("token %s is alredy used", token)
	}

	as.simpleTokens[token] = username
	as.simpleTokenKeeper.addSimpleToken(token)
	as.simpleTokensMu.Unlock()
	as.simpleTokenKeeper.tokensMu.Unlock()
}

func (as *authStore) invalidateUser(username string) {
	if as.simpleTokenKeeper == nil {
		return
	}
	as.simpleTokenKeeper.tokensMu.Lock()
	as.simpleTokensMu.Lock()
	for token, name := range as.simpleTokens {
		if strings.Compare(name, username) == 0 {
			delete(as.simpleTokens, token)
			as.simpleTokenKeeper.deleteSimpleToken(token)
		}
	}
	as.simpleTokensMu.Unlock()
	as.simpleTokenKeeper.tokensMu.Unlock()
}
