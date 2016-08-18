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

// Allows to use multiple urls
type URLProvider interface {
	// Return selected url
	Get() *url.URL
	// Select any other url, it is upto implementation to provide any
	// health checking/load balancing, it might be as simple as iterator
	Next() *url.URL
}

// Allows to iterate over urls, doesnt provide any health checking
type RoundRobinProvider struct {
	sync.RWMutex
	current int
	urls    []*url.URL
}

// Return selected url
func (p *RoundRobinProvider) Get() *url.URL {
	p.Lock()
	defer p.Unlock()
	return p.urls[p.current]
}

// Select next url if it exists, return selected url
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

// Return round robin provider with given urls
func NewRoundRobinProvider(urls ...*url.URL) *RoundRobinProvider {
	return &RoundRobinProvider{urls: urls}
}
