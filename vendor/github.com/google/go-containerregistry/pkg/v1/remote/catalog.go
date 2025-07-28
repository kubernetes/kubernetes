// Copyright 2019 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
)

type Catalogs struct {
	Repos []string `json:"repositories"`
	Next  string   `json:"next,omitempty"`
}

// CatalogPage calls /_catalog, returning the list of repositories on the registry.
func CatalogPage(target name.Registry, last string, n int, options ...Option) ([]string, error) {
	o, err := makeOptions(options...)
	if err != nil {
		return nil, err
	}

	f, err := newPuller(o).fetcher(o.context, target)
	if err != nil {
		return nil, err
	}

	uri := url.URL{
		Scheme:   target.Scheme(),
		Host:     target.RegistryStr(),
		Path:     "/v2/_catalog",
		RawQuery: fmt.Sprintf("last=%s&n=%d", url.QueryEscape(last), n),
	}

	req, err := http.NewRequest(http.MethodGet, uri.String(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := f.client.Do(req.WithContext(o.context))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if err := transport.CheckError(resp, http.StatusOK); err != nil {
		return nil, err
	}

	var parsed Catalogs
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}

	return parsed.Repos, nil
}

// Catalog calls /_catalog, returning the list of repositories on the registry.
func Catalog(ctx context.Context, target name.Registry, options ...Option) ([]string, error) {
	o, err := makeOptions(options...)
	if err != nil {
		return nil, err
	}

	// WithContext overrides the ctx passed directly.
	if o.context != context.Background() {
		ctx = o.context
	}

	return newPuller(o).catalog(ctx, target, o.pageSize)
}

func (f *fetcher) catalogPage(ctx context.Context, reg name.Registry, next string, pageSize int) (*Catalogs, error) {
	if next == "" {
		uri := &url.URL{
			Scheme: reg.Scheme(),
			Host:   reg.RegistryStr(),
			Path:   "/v2/_catalog",
		}
		if pageSize > 0 {
			uri.RawQuery = fmt.Sprintf("n=%d", pageSize)
		}
		next = uri.String()
	}

	req, err := http.NewRequestWithContext(ctx, "GET", next, nil)
	if err != nil {
		return nil, err
	}

	resp, err := f.client.Do(req)
	if err != nil {
		return nil, err
	}

	if err := transport.CheckError(resp, http.StatusOK); err != nil {
		return nil, err
	}

	parsed := Catalogs{}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}

	if err := resp.Body.Close(); err != nil {
		return nil, err
	}

	uri, err := getNextPageURL(resp)
	if err != nil {
		return nil, err
	}

	if uri != nil {
		parsed.Next = uri.String()
	}

	return &parsed, nil
}

type Catalogger struct {
	f        *fetcher
	reg      name.Registry
	pageSize int

	page *Catalogs
	err  error

	needMore bool
}

func (l *Catalogger) Next(ctx context.Context) (*Catalogs, error) {
	if l.needMore {
		l.page, l.err = l.f.catalogPage(ctx, l.reg, l.page.Next, l.pageSize)
	} else {
		l.needMore = true
	}
	return l.page, l.err
}

func (l *Catalogger) HasNext() bool {
	return l.page != nil && (!l.needMore || l.page.Next != "")
}
