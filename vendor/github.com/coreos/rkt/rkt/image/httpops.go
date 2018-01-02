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
	"fmt"
	"net/http"
	"net/url"
	"os"

	"github.com/coreos/rkt/rkt/config"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/hashicorp/errwrap"
)

// httpOps is a kind of facade around a downloader and a
// resumableSession. It provides some higher-level functions for
// fetching images and signature keys. It also is a provider of a
// remote fetcher for asc.
type httpOps struct {
	InsecureSkipTLSVerify bool
	S                     *imagestore.Store
	Headers               map[string]config.Headerer
	Debug                 bool
}

// DownloadSignature takes an asc instance and tries to get the
// signature. If the remote server asked to to defer the download,
// this function will return true and no error and no file.
func (o *httpOps) DownloadSignature(a *asc) (readSeekCloser, bool, error) {
	ensureLogger(o.Debug)
	diag.Printf("downloading signature from %v", a.Location)
	ascFile, err := a.Get()
	if err == nil {
		return ascFile, false, nil
	}
	if _, ok := err.(*statusAcceptedError); ok {
		log.Printf("server requested deferring the signature download")
		return NopReadSeekCloser(nil), true, nil
	}
	return nil, false, errwrap.Wrap(errors.New("error downloading the signature file"), err)
}

// DownloadSignatureAgain does a similar thing to DownloadSignature,
// but it expects the signature to be actually provided, that is - no
// deferring this time.
func (o *httpOps) DownloadSignatureAgain(a *asc) (readSeekCloser, error) {
	ensureLogger(o.Debug)
	ascFile, retry, err := o.DownloadSignature(a)
	if err != nil {
		return nil, err
	}
	if retry {
		return nil, fmt.Errorf("error downloading the signature file: server asked to defer the download again")
	}
	return ascFile, nil
}

// DownloadImage download the image, duh. It expects to actually
// receive the file, instead of being asked to use the cached version.
func (o *httpOps) DownloadImage(u *url.URL) (readSeekCloser, *cacheData, error) {
	ensureLogger(o.Debug)
	image, cd, err := o.DownloadImageWithETag(u, "")
	if err != nil {
		return nil, nil, err
	}
	if cd.UseCached {
		return nil, nil, fmt.Errorf("asked to use cached image even if not asked for that")
	}
	return image, cd, nil
}

// DownloadImageWithETag might download an image or tell you to use
// the cached image. In the latter case the returned file will be nil.
func (o *httpOps) DownloadImageWithETag(u *url.URL, etag string) (readSeekCloser, *cacheData, error) {
	var aciFile *removeOnClose // closed on error
	var errClose error         // error signaling to close aciFile

	ensureLogger(o.Debug)
	aciFile, err := getTmpROC(o.S, u.String())
	if err != nil {
		return nil, nil, err
	}

	defer func() {
		if errClose != nil {
			aciFile.Close()
		}
	}()

	session := o.getSession(u, aciFile.File, "ACI", etag)
	dl := o.getDownloader(session)
	errClose = dl.Download(u, aciFile.File)
	if errClose != nil {
		return nil, nil, errwrap.Wrap(errors.New("error downloading ACI"), errClose)
	}

	if session.Cd.UseCached {
		aciFile.Close()
		return NopReadSeekCloser(nil), session.Cd, nil
	}

	return aciFile, session.Cd, nil
}

// AscRemoteFetcher provides a remoteAscFetcher for asc.
func (o *httpOps) AscRemoteFetcher() *remoteAscFetcher {
	ensureLogger(o.Debug)
	f := func(u *url.URL, file *os.File) error {
		switch u.Scheme {
		case "http", "https":
		default:
			return fmt.Errorf("invalid signature location: expected %q scheme, got %q", "http(s)", u.Scheme)
		}
		session := o.getSession(u, file, "signature", "")
		dl := o.getDownloader(session)
		err := dl.Download(u, file)
		if err != nil {
			return err
		}
		if session.Cd.UseCached {
			return fmt.Errorf("unexpected cache reuse request for signature %q", u.String())
		}
		return nil
	}
	return &remoteAscFetcher{
		F: f,
		S: o.S,
	}
}

func (o *httpOps) getSession(u *url.URL, file *os.File, label, etag string) *resumableSession {
	eTagFilePath := fmt.Sprintf("%s.etag", file.Name())
	return &resumableSession{
		InsecureSkipTLSVerify: o.InsecureSkipTLSVerify,
		Headers:               o.getHeaders(u, etag),
		Headerers:             o.Headers,
		File:                  file,
		ETagFilePath:          eTagFilePath,
		Label:                 label,
	}
}

func (o *httpOps) getDownloader(session downloadSession) *downloader {
	return &downloader{
		Session: session,
	}
}

func (o *httpOps) getHeaders(u *url.URL, etag string) http.Header {
	options := o.getHeadersForURL(u, etag)
	if etag != "" {
		options.Add("If-None-Match", etag)
	}
	return options
}

func (o *httpOps) getHeadersForURL(u *url.URL, etag string) http.Header {
	return make(http.Header)
}
