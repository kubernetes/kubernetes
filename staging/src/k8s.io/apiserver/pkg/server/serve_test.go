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

package server

import (
	"fmt"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func (i *IPBasedLimit) decrementIP(conn, sourceIP string) {
	i.mutex.Lock()
	defer i.mutex.Unlock()
	i.decrementIPLocked(conn, sourceIP)
}

func TestBasicIPLimit(t *testing.T) {
	var ipLimit IPBasedLimit
	validateIPLimitState(t, &ipLimit, "Basic/raw IPBasedLimit", 0, 0, 0, 0)
	ipLimit.incrementIP("1.2.3.4:1234", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Basic/raw IPBasedLimit increment", 0, 0, 0, 0)
	ipLimit.decrementIP("1.2.3.4:1235", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Basic/raw IPBasedLimit decrement", 0, 0, 0, 0)
	ipLimit.incrementIP("127.0.0.1:99999", "127.0.0.1")
	validateIPLimitState(t, &ipLimit, "Basic/raw IPBasedLimit increment localhost", 0, 0, 0, 0)
	ipLimit.decrementIP("127.0.0.1:1", "127.0.0.1")
	validateIPLimitState(t, &ipLimit, "Basic/raw IPBasedLimit decrement localhost", 0, 0, 0, 0)
}

func TestIPLimitInitialization(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 10, Max: 20}
	validateIPLimitState(t, &ipLimit, "IPBasedLimit explicit initialization", 0, 10, 0, 20)
}

func TestIPLimitWithLocalHost(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 10, Max: 20}
	ipLimit.incrementIP("127.0.0.1:99999", "127.0.0.1")
	validateIPLimitState(t, &ipLimit, "IPv4 localhost increment call for IPBasedLimit", 0, 10, 0, 20)
	ipLimit.decrementIP("127.0.0.1:1", "127.0.0.1")
	validateIPLimitState(t, &ipLimit, "IPv4 localhost decrement call for IPBasedLimit", 0, 10, 0, 20)
	ipv6loopback := net.IPv6loopback.String()
	ipLimit.incrementIP(ipv6loopback+":42", ipv6loopback)
	validateIPLimitState(t, &ipLimit, "IPv6 localhost increment call for IPBasedLimit", 0, 10, 0, 20)
	ipLimit.decrementIP(ipv6loopback+":42", ipv6loopback)
	validateIPLimitState(t, &ipLimit, "IPv6 localhost decrement call for IPBasedLimit", 0, 10, 0, 20)
}

func TestIPLimitIncrementAndDecrement(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 10, Max: 20}
	ipLimit.incrementIP("1.2.3.4:1234", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "First increment call for IPBasedLimit", 1, 10, 1, 20)
	val, ok := ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)
	ipLimit.incrementIP("1.2.3.4:1235", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Second increment call for IPBasedLimit", 1, 10, 2, 20)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.incrementIP("4.3.2.1:111", "4.3.2.1")
	validateIPLimitState(t, &ipLimit, "First increment call for IPBasedLimit", 2, 10, 3, 20)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)
	ipLimit.incrementIP("4.3.2.1:222", "4.3.2.1")
	validateIPLimitState(t, &ipLimit, "Second increment call for IPBasedLimit", 2, 10, 4, 20)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("4.3.2.1:111", "4.3.2.1")
	validateIPLimitState(t, &ipLimit, "First decrement call for IPBasedLimit", 2, 10, 3, 20)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.True(t, ok, "Decrement call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("4.3.2.1:222", "4.3.2.1")
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 1, 10, 2, 20)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.EqualValues(t, val, 0, "Deleting decrement call for IPBasedLimit has a val of %d", 0)
	assert.False(t, ok, "Deleting decrement call for IPBasedLimit had a val")
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Unmodified ip in IPBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("1.2.3.4:1235", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "First decrement call for IPBasedLimit", 1, 10, 1, 20)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("1.2.3.4:1234", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 0, 10, 0, 20)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.False(t, ok, "Deleting decrement call for IPBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for IPBasedLimit has a val of %d", 0)
}

func TestIPLimitIncrementFailOnIPLimit(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 3, Max: 6}
	err := ipLimit.incrementIP("1.2.3.4:1234", "1.2.3.4")
	assert.NoError(t, err)
	validateIPLimitState(t, &ipLimit, "First increment call for IPBasedLimit", 1, 3, 1, 6)
	val, ok := ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	err = ipLimit.incrementIP("1.2.3.4:1235", "1.2.3.4")
	assert.NoError(t, err)
	validateIPLimitState(t, &ipLimit, "Second increment call for IPBasedLimit", 1, 3, 2, 6)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.EqualValues(t, val, 2, "Post increment IPBasedLimit has a Max of %d", val)

	err = ipLimit.incrementIP("1.2.3.4:1236", "1.2.3.4")
	assert.NoError(t, err)
	validateIPLimitState(t, &ipLimit, "Third increment call for IPBasedLimit", 1, 3, 3, 6)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 3, "Post increment IPBasedLimit has a Max of %d", val)

	err = ipLimit.incrementIP("1.2.3.4:1234", "1.2.3.4")
	assert.Error(t, err)
	if limErr, ok := err.(IPLimitExceededError); ok {
		assert.True(t, limErr.Timeout(), "IPLimit IPLimitExceededError should return timeout true")
		assert.True(t, limErr.Temporary(), "IPLimit IPLimitExceededError should return temporary true")
	} else {
		assert.Fail(t, "Expected a IPLimitExceededError from incrementIP fail but got %T", err)
	}
	validateIPLimitState(t, &ipLimit, "Second increment call for IPBasedLimit", 1, 3, 3, 6)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 3, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("1.2.3.4:1234", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Decrement call for IPBasedLimit", 1, 3, 2, 6)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("1.2.3.4:1235", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Decrement call for IPBasedLimit", 1, 3, 1, 6)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("1.2.3.4:1236", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 0, 3, 0, 6)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.False(t, ok, "Deleting decrement call for IPBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for IPBasedLimit has a val of %d", 0)
}

func TestIPLimitIncrementFailOnTotalLimit(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 2, Max: 3}
	err := ipLimit.incrementIP("1.2.3.4:1234", "1.2.3.4")
	assert.NoError(t, err)
	validateIPLimitState(t, &ipLimit, "First increment call for IPBasedLimit", 1, 2, 1, 3)
	val, ok := ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	err = ipLimit.incrementIP("1.2.3.4:1235", "1.2.3.4")
	assert.NoError(t, err)
	validateIPLimitState(t, &ipLimit, "Second increment call for IPBasedLimit", 1, 2, 2, 3)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment IPBasedLimit has a Max of %d", val)

	err = ipLimit.incrementIP("4.3.2.1:111", "4.3.2.1")
	assert.NoError(t, err)
	validateIPLimitState(t, &ipLimit, "First increment call for second ip IPBasedLimit", 2, 2, 3, 3)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	err = ipLimit.incrementIP("4.3.2.1:222", "4.3.2.1")
	assert.Error(t, err)
	if limErr, ok := err.(IPLimitExceededError); ok {
		assert.True(t, limErr.Timeout(), "IPLimit IPLimitExceededError should return timeout true")
		assert.True(t, limErr.Temporary(), "IPLimit IPLimitExceededError should return temporary true")
	} else {
		assert.Fail(t, "Expected a IPLimitExceededError from incrementIP fail but got %T", err)
	}
	validateIPLimitState(t, &ipLimit, "Second increment call for IPBasedLimit", 2, 2, 3, 3)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("1.2.3.4:1235", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Decrement call for IPBasedLimit", 2, 2, 2, 3)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("1.2.3.4:1234", "1.2.3.4")
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 1, 2, 1, 3)
	val, ok = ipLimit.Limits["1.2.3.4"]
	assert.False(t, ok, "Deleting decrement call for IPBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for IPBasedLimit has a val of %d", 0)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.True(t, ok, "Unmodified IP for IPBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Unmodified IP for IPBasedLimit has a Max of %d", val)

	ipLimit.decrementIP("4.3.2.1:111", "4.3.2.1")
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 0, 2, 0, 3)
	val, ok = ipLimit.Limits["4.3.2.1"]
	assert.False(t, ok, "Deleting decrement call for IPBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for IPBasedLimit has a val of %d", 0)
}

func TestParallelIPLimitAccessNoLimit(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 10, Max: 100}
	stop := time.Now().Add(time.Second * 10) // Long enough to prevent flaky ???
	stateHolder := parallelStateHolder{ipLimit: &ipLimit, waitSize: 100,
		stop: stop, good: 0, limit: 0, count: 0}
	ips := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5",
		"6.6.6.6", "7.7.7.7", "8.8.8.8", "9.9.9.9",
		"1973:dead:beef:0121:1973:dead:beef:0121"}
	done := make(chan struct{}, 100)
	defer close(done)
	for ctr, ip := range ips {
		for i := 0; i < 10; i++ {
			go stateHolder.accessIPLimit(ip, ctr, done)
		}
	}
	for i := 0; i < 100; i++ {
		<-done
	}
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 0, 10, 0, 100)
	assert.EqualValues(t, stateHolder.good, 100, "Concurrent increment count for IPBasedLimit %d successes", 0)
	assert.EqualValues(t, stateHolder.limit, 0, "Concurrent increment count for IPBasedLimit %d Limits", 0)
}

func TestParallelIPLimitHitIPLimit(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 8, Max: 100}
	stop := time.Now().Add(time.Second * 10) // Long enough to prevent flaky ???
	stateHolder := parallelStateHolder{ipLimit: &ipLimit, waitSize: 100,
		stop: stop, good: 0, limit: 0, count: 0}
	ips := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5",
		"6.6.6.6", "7.7.7.7", "8.8.8.8", "9.9.9.9",
		"1973:dead:beef:0121:1973:dead:beef:0121"}
	done := make(chan struct{}, 100)
	defer close(done)
	for ctr, ip := range ips {
		for i := 0; i < 10; i++ {
			go stateHolder.accessIPLimit(ip, ctr, done)
		}
	}
	for i := 0; i < 100; i++ {
		<-done
	}
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 0, 8, 0, 100)
	assert.EqualValues(t, stateHolder.good, 80, "Concurrent increment count for IPBasedLimit %d successes", 0)
	assert.EqualValues(t, stateHolder.limit, 20, "Concurrent increment count for IPBasedLimit %d Limits", 0)
}

func TestParallelIPLimitHitTotalLimit(t *testing.T) {
	ipLimit := IPBasedLimit{Limits: make(map[string]int), Decrementers: make(map[string]func()), MaxPerIP: 10, Max: 79}
	stop := time.Now().Add(time.Second * 10) // Long enough to prevent flaky ???
	stateHolder := parallelStateHolder{ipLimit: &ipLimit, waitSize: 100,
		stop: stop, good: 0, limit: 0, count: 0}
	ips := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5",
		"6.6.6.6", "7.7.7.7", "8.8.8.8", "9.9.9.9",
		"1977:dead:beef:0525:2015:dead:beef:1218"}
	done := make(chan struct{}, 100)
	defer close(done)
	for ctr, ip := range ips {
		for i := 0; i < 10; i++ {
			go stateHolder.accessIPLimit(ip, ctr, done)
		}
	}
	for i := 0; i < 100; i++ {
		<-done
	}
	validateIPLimitState(t, &ipLimit, "Deleting decrement call for IPBasedLimit", 0, 10, 0, 79)
	assert.EqualValues(t, stateHolder.good, 79, "Concurrent increment count for IPBasedLimit %d successes", 0)
	assert.EqualValues(t, stateHolder.limit, 21, "Concurrent increment count for IPBasedLimit %d Limits", 0)
}

func (sh *parallelStateHolder) accessIPLimit(ip string, ctr int, done chan<- struct{}) {
	defer func() { done <- struct{}{} }()
	err := sh.ipLimit.incrementIP(fmt.Sprintf("%s:%d", ip, ctr), ip)
	if err == nil {
		atomic.AddUint64(&sh.good, 1)
	} else {
		atomic.AddUint64(&sh.limit, 1)
	}
	atomic.AddUint64(&sh.count, 1)
	if err == nil {
		sh.wait()
		sh.ipLimit.decrementIP(fmt.Sprintf("%s:%d", ip, ctr), ip)
	}
}

func (sh *parallelStateHolder) wait() {
	var current uint64
	for current < sh.waitSize {
		time.Sleep(time.Millisecond * 10)
		if time.Now().After(sh.stop) {
			break
		}
		current = atomic.LoadUint64(&sh.count)
	}
}

type parallelStateHolder struct {
	ipLimit  *IPBasedLimit
	waitSize uint64
	stop     time.Time
	good     uint64
	limit    uint64
	count    uint64
}

func validateIPLimitState(t *testing.T, ipLimit *IPBasedLimit, base string, expMapSize, expIpMax, expTtl, expMax int) {
	assert.EqualValues(t, len(ipLimit.Limits), expMapSize, "%s map is size %d not %d", base, len(ipLimit.Limits), expMapSize)
	assert.EqualValues(t, ipLimit.MaxPerIP, expIpMax, "%s MaxPerIP is %d not %d", base, ipLimit.MaxPerIP, expIpMax)
	assert.EqualValues(t, ipLimit.total, expTtl, "%s count is %d not %d", base, ipLimit.total, expTtl)
	assert.EqualValues(t, ipLimit.Max, expMax, "%s Max is %d not %d", base, ipLimit.Max, expMax)
}
