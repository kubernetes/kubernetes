// Copyright 2015 The rkt Authors
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

package image

import (
	"errors"
	"net/url"
	"os"

	"github.com/coreos/rkt/store"
	"github.com/hashicorp/errwrap"
)

// ascFetcher is an interface used by asc to get the desired signature
// file.
type ascFetcher interface {
	// Get fetches the file from passed location.
	Get(location string) (readSeekCloser, error)
}

// localAscFetcher is an implementation of ascFetcher getting
// signature files from a local filesystem.
type localAscFetcher struct{}

func (*localAscFetcher) Get(location string) (readSeekCloser, error) {
	return os.Open(location)
}

// remoteAscFetcher is an implementation of ascFetcher getting
// signature files from remote locations.
type remoteAscFetcher struct {
	// F is a function that actually does the fetching
	F func(*url.URL, *os.File) error
	// S is a store - used for getting a temporary file
	S *store.Store
}

func (f *remoteAscFetcher) Get(location string) (readSeekCloser, error) {
	roc, err := getTmpROC(f.S, location)
	if err != nil {
		return nil, err
	}
	defer func() { maybeClose(roc) }()

	u, err := url.Parse(location)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("invalid signature location"), err)
	}
	if err := f.F(u, roc.File); err != nil {
		return nil, err
	}
	retRoc := roc
	roc = nil
	return retRoc, nil
}

// asc is an abstraction for getting signature files.
type asc struct {
	// Location is a string passed to the Fetcher.
	Location string
	// Fetcher (if available) does the actual fetching of a
	// signature key.
	Fetcher ascFetcher
}

// Get fetches a signature file. It returns nil and no error if there
// was no fetcher set.
func (a *asc) Get() (readSeekCloser, error) {
	if a.Fetcher != nil {
		return a.Fetcher.Get(a.Location)
	}
	return nil, nil
}
