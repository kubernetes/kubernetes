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
	"io"
	"net/url"
	"time"

	rktflag "github.com/coreos/rkt/rkt/flag"

	"github.com/coreos/rkt/pkg/keystore"
	"github.com/coreos/rkt/rkt/config"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/hashicorp/errwrap"
)

// httpFetcher is used to download images from http or https URLs.
type httpFetcher struct {
	InsecureFlags *rktflag.SecFlags
	S             *imagestore.Store
	Ks            *keystore.Keystore
	Rem           *imagestore.Remote
	NoCache       bool
	Debug         bool
	Headers       map[string]config.Headerer
}

// Hash fetches the URL, optionally verifies it against passed asc,
// stores it in the store and returns the hash.
func (f *httpFetcher) Hash(u *url.URL, a *asc) (string, error) {
	ensureLogger(f.Debug)
	urlStr := u.String()

	if !f.NoCache && f.Rem != nil {
		if useCached(f.Rem.DownloadTime, f.Rem.CacheMaxAge) {
			diag.Printf("image for %s isn't expired, not fetching.", urlStr)
			return f.Rem.BlobKey, nil
		}
	}

	diag.Printf("fetching image from %s", urlStr)

	aciFile, cd, err := f.fetchURL(u, a, eTag(f.Rem))
	if err != nil {
		return "", err
	}
	defer aciFile.Close()

	if key := maybeUseCached(f.Rem, cd); key != "" {
		// TODO(krnowak): that does not update the store with
		// the new CacheMaxAge and Download Time, so it will
		// query the server every time after initial
		// CacheMaxAge is exceeded
		return key, nil
	}
	key, err := f.S.WriteACI(aciFile, imagestore.ACIFetchInfo{
		Latest: false,
	})
	if err != nil {
		return "", err
	}

	// TODO(krnowak): What's the point of the second parameter?
	// The SigURL field in imagestore.Remote seems to be completely
	// unused.
	newRem := imagestore.NewRemote(urlStr, a.Location)
	newRem.BlobKey = key
	newRem.DownloadTime = time.Now()
	if cd != nil {
		newRem.ETag = cd.ETag
		newRem.CacheMaxAge = cd.MaxAge
	}
	err = f.S.WriteRemote(newRem)
	if err != nil {
		return "", err
	}

	return key, nil
}

func (f *httpFetcher) fetchURL(u *url.URL, a *asc, etag string) (readSeekCloser, *cacheData, error) {
	if f.InsecureFlags.SkipTLSCheck() && f.Ks != nil {
		log.Printf("warning: TLS verification has been disabled")
	}
	if f.InsecureFlags.SkipImageCheck() && f.Ks != nil {
		log.Printf("warning: image signature verification has been disabled")
	}

	if f.InsecureFlags.SkipImageCheck() || f.Ks == nil {
		o := f.httpOps()
		aciFile, cd, err := o.DownloadImageWithETag(u, etag)
		if err != nil {
			return nil, nil, err
		}
		return aciFile, cd, err
	}

	return f.fetchVerifiedURL(u, a, etag)
}

func (f *httpFetcher) fetchVerifiedURL(u *url.URL, a *asc, etag string) (readSeekCloser, *cacheData, error) {
	var aciFile readSeekCloser // closed on error
	var errClose error         // error signaling to close aciFile

	o := f.httpOps()
	f.maybeOverrideAscFetcherWithRemote(o, u, a)
	ascFile, retry, err := o.DownloadSignature(a)
	if err != nil {
		return nil, nil, err
	}
	defer ascFile.Close()

	aciFile, cd, err := o.DownloadImageWithETag(u, etag)
	if err != nil {
		return nil, nil, err
	}

	defer func() {
		if errClose != nil {
			aciFile.Close()
		}
	}()

	if cd.UseCached {
		aciFile.Close()
		return NopReadSeekCloser(nil), cd, nil
	}

	if retry {
		ascFile.Close()
		ascFile, errClose = o.DownloadSignatureAgain(a)
		if errClose != nil {
			ascFile = NopReadSeekCloser(nil)
			return nil, nil, errClose
		}
	}

	errClose = f.validate(aciFile, ascFile)
	if errClose != nil {
		return nil, nil, errClose
	}

	return aciFile, cd, nil
}

func (f *httpFetcher) httpOps() *httpOps {
	return &httpOps{
		InsecureSkipTLSVerify: f.InsecureFlags.SkipTLSCheck(),
		S:       f.S,
		Headers: f.Headers,
		Debug:   f.Debug,
	}
}

func (f *httpFetcher) validate(aciFile, ascFile io.ReadSeeker) error {
	v, err := newValidator(aciFile)
	if err != nil {
		return err
	}
	entity, err := v.ValidateWithSignature(f.Ks, ascFile)
	if err != nil {
		return err
	}
	if _, err := aciFile.Seek(0, 0); err != nil {
		return errwrap.Wrap(errors.New("error seeking ACI file"), err)
	}

	printIdentities(entity)
	return nil
}

func (f *httpFetcher) maybeOverrideAscFetcherWithRemote(o *httpOps, u *url.URL, a *asc) {
	if a.Fetcher != nil {
		return
	}
	u2 := ascURLFromImgURL(u)
	a.Location = u2.String()
	a.Fetcher = o.AscRemoteFetcher()
}
