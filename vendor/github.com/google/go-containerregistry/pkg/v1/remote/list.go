// Copyright 2018 Google LLC All Rights Reserved.
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
	"strings"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
)

// ListWithContext calls List with the given context.
//
// Deprecated: Use List and WithContext. This will be removed in a future release.
func ListWithContext(ctx context.Context, repo name.Repository, options ...Option) ([]string, error) {
	return List(repo, append(options, WithContext(ctx))...)
}

// List calls /tags/list for the given repository, returning the list of tags
// in the "tags" property.
func List(repo name.Repository, options ...Option) ([]string, error) {
	o, err := makeOptions(options...)
	if err != nil {
		return nil, err
	}
	return newPuller(o).List(o.context, repo)
}

type Tags struct {
	Name string   `json:"name"`
	Tags []string `json:"tags"`
	Next string   `json:"next,omitempty"`
}

func (f *fetcher) listPage(ctx context.Context, repo name.Repository, next string, pageSize int) (*Tags, error) {
	if next == "" {
		uri := &url.URL{
			Scheme: repo.Scheme(),
			Host:   repo.RegistryStr(),
			Path:   fmt.Sprintf("/v2/%s/tags/list", repo.RepositoryStr()),
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

	parsed := Tags{}
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

// getNextPageURL checks if there is a Link header in a http.Response which
// contains a link to the next page. If yes it returns the url.URL of the next
// page otherwise it returns nil.
func getNextPageURL(resp *http.Response) (*url.URL, error) {
	link := resp.Header.Get("Link")
	if link == "" {
		return nil, nil
	}

	if link[0] != '<' {
		return nil, fmt.Errorf("failed to parse link header: missing '<' in: %s", link)
	}

	end := strings.Index(link, ">")
	if end == -1 {
		return nil, fmt.Errorf("failed to parse link header: missing '>' in: %s", link)
	}
	link = link[1:end]

	linkURL, err := url.Parse(link)
	if err != nil {
		return nil, err
	}
	if resp.Request == nil || resp.Request.URL == nil {
		return nil, nil
	}
	linkURL = resp.Request.URL.ResolveReference(linkURL)
	return linkURL, nil
}

type Lister struct {
	f        *fetcher
	repo     name.Repository
	pageSize int

	page *Tags
	err  error

	needMore bool
}

func (l *Lister) Next(ctx context.Context) (*Tags, error) {
	if l.needMore {
		l.page, l.err = l.f.listPage(ctx, l.repo, l.page.Next, l.pageSize)
	} else {
		l.needMore = true
	}
	return l.page, l.err
}

func (l *Lister) HasNext() bool {
	return l.page != nil && (!l.needMore || l.page.Next != "")
}
