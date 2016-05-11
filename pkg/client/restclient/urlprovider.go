/*
Copyright 2016 The Kubernetes Authors.

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

package restclient

import (
	"net/url"
	"sync"
)

// A URLProvider allows to use multiple URLs.
// Also maintains state of currently working URL.
type URLProvider interface {
	// Get returns currently selected URL.
	Get() *url.URL
	// Next selects any other URL, it is upto implementation to provide any
	// health checking/load balancing, it might be as simple as an iterator.
	Next() *url.URL
	// WrapWithRetry executes function until success or until all urls were visited.
	WrapWithRetry(func() error) error
}

// A RoundRobinProvider allows to iterate over URLs, doesnt provide any health checking.
type RoundRobinProvider struct {
	sync.Mutex
	current int
	urls    []*url.URL
}

// Get returns currently selected url.
func (p *RoundRobinProvider) Get() *url.URL {
	p.Lock()
	defer p.Unlock()
	return p.urls[p.current]
}

// Next selects next URL if it exists, returns selected URL.
func (p *RoundRobinProvider) Next() *url.URL {
	p.Lock()
	defer p.Unlock()
	if p.current >= len(p.urls)-1 {
		p.current = 0
	} else {
		p.current++
	}
	return p.urls[p.current]
}

// WrapWithRetry executes any function until success or until all urls will be visited.
func (p *RoundRobinProvider) WrapWithRetry(f func() error) error {
	var err error
	for range p.urls {
		err = f()
		if err != nil {
			p.Next()
		} else {
			return nil
		}
	}
	return err
}

// NewRoundRobinProvider returns pointer to new RoundRobinProvider with passed URLs.
func NewRoundRobinProvider(urls ...*url.URL) *RoundRobinProvider {
	return &RoundRobinProvider{urls: urls}
}
