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
	"time"
)

const (
	letters                  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	defaultSimpleTokenLength = 16
	simpleTokenTTL           = 5 * time.Minute
	simpleTokenTTLResolution = 1 * time.Second
)

type simpleTokenTTLKeeper struct {
	tokens              map[string]time.Time
	addSimpleTokenCh    chan string
	resetSimpleTokenCh  chan string
	deleteSimpleTokenCh chan string
	stopCh              chan chan struct{}
	deleteTokenFunc     func(string)
}

func NewSimpleTokenTTLKeeper(deletefunc func(string)) *simpleTokenTTLKeeper {
	stk := &simpleTokenTTLKeeper{
		tokens:              make(map[string]time.Time),
		addSimpleTokenCh:    make(chan string, 1),
		resetSimpleTokenCh:  make(chan string, 1),
		deleteSimpleTokenCh: make(chan string, 1),
		stopCh:              make(chan chan struct{}),
		deleteTokenFunc:     deletefunc,
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
	tm.addSimpleTokenCh <- token
}

func (tm *simpleTokenTTLKeeper) resetSimpleToken(token string) {
	tm.resetSimpleTokenCh <- token
}

func (tm *simpleTokenTTLKeeper) deleteSimpleToken(token string) {
	tm.deleteSimpleTokenCh <- token
}
func (tm *simpleTokenTTLKeeper) run() {
	tokenTicker := time.NewTicker(simpleTokenTTLResolution)
	defer tokenTicker.Stop()
	for {
		select {
		case t := <-tm.addSimpleTokenCh:
			tm.tokens[t] = time.Now().Add(simpleTokenTTL)
		case t := <-tm.resetSimpleTokenCh:
			if _, ok := tm.tokens[t]; ok {
				tm.tokens[t] = time.Now().Add(simpleTokenTTL)
			}
		case t := <-tm.deleteSimpleTokenCh:
			delete(tm.tokens, t)
		case <-tokenTicker.C:
			nowtime := time.Now()
			for t, tokenendtime := range tm.tokens {
				if nowtime.After(tokenendtime) {
					tm.deleteTokenFunc(t)
					delete(tm.tokens, t)
				}
			}
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
	as.simpleTokensMu.Lock()

	_, ok := as.simpleTokens[token]
	if ok {
		plog.Panicf("token %s is alredy used", token)
	}

	as.simpleTokens[token] = username
	as.simpleTokenKeeper.addSimpleToken(token)
	as.simpleTokensMu.Unlock()
}

func (as *authStore) invalidateUser(username string) {
	as.simpleTokensMu.Lock()
	defer as.simpleTokensMu.Unlock()

	for token, name := range as.simpleTokens {
		if strings.Compare(name, username) == 0 {
			delete(as.simpleTokens, token)
			as.simpleTokenKeeper.deleteSimpleToken(token)
		}
	}
}
