/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

type URLProvider interface {
	Get() *url.URL
	Next() *url.URL
}

type RoundRobinProvider struct {
	sync.RWMutex
	urls    []*url.URL
	current int
}

func (p *RoundRobinProvider) Get() *url.URL {
	p.RLock()
	defer p.RUnlock()
	return p.urls[p.current]
}

func (p *RoundRobinProvider) Next() *url.URL {
	p.RLock()
	defer p.RUnlock()
	if p.current >= len(p.urls)-1 {
		p.current = 0
	} else {
		p.current++
	}
	return p.Get()
}
