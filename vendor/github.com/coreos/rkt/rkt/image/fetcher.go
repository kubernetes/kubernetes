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
	"container/list"
	"fmt"
	"net/url"
	"runtime"

	"github.com/coreos/rkt/common/apps"
	"github.com/coreos/rkt/stage0"
	"github.com/coreos/rkt/store"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/discovery"
	"github.com/appc/spec/schema/types"
)

// Fetcher will try to fetch images into the store.
type Fetcher action

// FetchImage will take an image as either a path, a URL or a name
// string and import it into the store if found. If ascPath is not "",
// it must exist as a local file and will be used as the signature
// file for verification, unless verification is disabled. If
// f.WithDeps is true also image dependencies are fetched.
func (f *Fetcher) FetchImage(img string, ascPath string, imgType apps.AppImageType) (string, error) {
	ensureLogger(f.Debug)
	a := f.getAsc(ascPath)
	hash, err := f.fetchSingleImage(img, a, imgType)
	if err != nil {
		return "", err
	}
	if f.WithDeps {
		err = f.fetchImageDeps(hash)
		if err != nil {
			return "", err
		}
	}
	return hash, nil
}

func (f *Fetcher) getAsc(ascPath string) *asc {
	if ascPath != "" {
		return &asc{
			Location: ascPath,
			Fetcher:  &localAscFetcher{},
		}
	}
	return &asc{}
}

// fetchImageDeps will recursively fetch all the image dependencies
func (f *Fetcher) fetchImageDeps(hash string) error {
	imgsl := list.New()
	seen := map[string]struct{}{}
	f.addImageDeps(hash, imgsl, seen)
	for el := imgsl.Front(); el != nil; el = el.Next() {
		a := &asc{}
		img := el.Value.(string)
		hash, err := f.fetchSingleImage(img, a, apps.AppImageName)
		if err != nil {
			return err
		}
		f.addImageDeps(hash, imgsl, seen)
	}
	return nil
}

func (f *Fetcher) addImageDeps(hash string, imgsl *list.List, seen map[string]struct{}) error {
	dependencies, err := f.getImageDeps(hash)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("failed to get dependencies for image ID %q", hash), err)
	}
	for _, d := range dependencies {
		imgName := d.ImageName.String()
		app, err := discovery.NewApp(imgName, d.Labels.ToMap())
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("one of image ID's %q dependencies (image %q) is invalid", hash, imgName), err)
		}
		appStr := app.String()
		if _, ok := seen[appStr]; ok {
			continue
		}
		imgsl.PushBack(app.String())
		seen[appStr] = struct{}{}
	}
	return nil
}

func (f *Fetcher) getImageDeps(hash string) (types.Dependencies, error) {
	key, err := f.S.ResolveKey(hash)
	if err != nil {
		return nil, err
	}
	im, err := f.S.GetImageManifest(key)
	if err != nil {
		return nil, err
	}
	return im.Dependencies, nil
}

func (f *Fetcher) fetchSingleImage(img string, a *asc, imgType apps.AppImageType) (string, error) {
	if imgType == apps.AppImageGuess {
		imgType = guessImageType(img)
	}
	if imgType == apps.AppImageHash {
		return "", fmt.Errorf("cannot fetch a hash '%q', expected either a URL, a path or an image name", img)
	}

	switch imgType {
	case apps.AppImageURL:
		return f.fetchSingleImageByURL(img, a)
	case apps.AppImagePath:
		return f.fetchSingleImageByPath(img, a)
	case apps.AppImageName:
		return f.fetchSingleImageByName(img, a)
	default:
		return "", fmt.Errorf("unknown image type %d", imgType)
	}
}

type remoteCheck int

const (
	remoteCheckLax remoteCheck = iota
	remoteCheckStrict
)

func (f *Fetcher) fetchSingleImageByURL(urlStr string, a *asc) (string, error) {
	u, err := url.Parse(urlStr)
	if err != nil {
		return "", errwrap.Wrap(fmt.Errorf("invalid image URL %q", urlStr), err)
	}

	switch u.Scheme {
	case "http", "https":
		return f.fetchSingleImageByHTTPURL(u, a)
	case "docker":
		return f.fetchSingleImageByDockerURL(u)
	case "file":
		return f.fetchSingleImageByPath(u.Path, a)
	case "":
		return "", fmt.Errorf("expected image URL %q to contain a scheme", urlStr)
	default:
		return "", fmt.Errorf("an unsupported URL scheme %q - the only URL schemes supported by rkt are docker, http, https and file", u.Scheme)
	}
}

func (f *Fetcher) fetchSingleImageByHTTPURL(u *url.URL, a *asc) (string, error) {
	rem, err := f.getRemoteForURL(u)
	if err != nil {
		return "", err
	}
	if h := f.maybeCheckRemoteFromStore(rem, remoteCheckStrict); h != "" {
		return h, nil
	}
	if h, err := f.maybeFetchHTTPURLFromRemote(rem, u, a); h != "" || err != nil {
		return h, err
	}
	return "", fmt.Errorf("unable to fetch image from URL %q: either image was not found in the store or store was disabled and fetching from remote yielded nothing or it was disabled", u.String())
}

func (f *Fetcher) fetchSingleImageByDockerURL(u *url.URL) (string, error) {
	rem, err := f.getRemoteForURL(u)
	if err != nil {
		return "", err
	}
	// TODO(krnowak): use strict checking when we implement
	// setting CacheMaxAge in store.Remote for docker images
	if h := f.maybeCheckRemoteFromStore(rem, remoteCheckLax); h != "" {
		return h, nil
	}
	if h, err := f.maybeFetchDockerURLFromRemote(u); h != "" || err != nil {
		return h, err
	}
	return "", fmt.Errorf("unable to fetch docker image from URL %q: either image was not found in the store or store was disabled and fetching from remote yielded nothing or it was disabled", u.String())
}

func (f *Fetcher) getRemoteForURL(u *url.URL) (*store.Remote, error) {
	if f.NoCache {
		return nil, nil
	}
	urlStr := u.String()
	if rem, ok, err := f.S.GetRemote(urlStr); err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("failed to fetch URL %q", urlStr), err)
	} else if ok {
		return rem, nil
	}
	return nil, nil
}

func (f *Fetcher) maybeCheckRemoteFromStore(rem *store.Remote, check remoteCheck) string {
	if f.NoStore || rem == nil {
		return ""
	}
	useBlobKey := false
	switch check {
	case remoteCheckLax:
		useBlobKey = true
	case remoteCheckStrict:
		useBlobKey = useCached(rem.DownloadTime, rem.CacheMaxAge)
	}
	if useBlobKey {
		log.Printf("using image from local store for url %s", rem.ACIURL)
		return rem.BlobKey
	}
	return ""
}

func (f *Fetcher) maybeFetchHTTPURLFromRemote(rem *store.Remote, u *url.URL, a *asc) (string, error) {
	if !f.StoreOnly {
		log.Printf("remote fetching from URL %q", u.String())
		hf := &httpFetcher{
			InsecureFlags: f.InsecureFlags,
			S:             f.S,
			Ks:            f.Ks,
			Rem:           rem,
			Debug:         f.Debug,
			Headers:       f.Headers,
		}
		return hf.GetHash(u, a)
	}
	return "", nil
}

func (f *Fetcher) maybeFetchDockerURLFromRemote(u *url.URL) (string, error) {
	if !f.StoreOnly {
		log.Printf("remote fetching from URL %q", u.String())
		df := &dockerFetcher{
			InsecureFlags: f.InsecureFlags,
			DockerAuth:    f.DockerAuth,
			S:             f.S,
			Debug:         f.Debug,
		}
		return df.GetHash(u)
	}
	return "", nil
}

func (f *Fetcher) fetchSingleImageByPath(path string, a *asc) (string, error) {
	log.Printf("using image from file %s", path)
	ff := &fileFetcher{
		InsecureFlags: f.InsecureFlags,
		S:             f.S,
		Ks:            f.Ks,
		Debug:         f.Debug,
	}
	return ff.GetHash(path, a)
}

type appBundle struct {
	App *discovery.App
	Str string
}

func newAppBundle(name string) (*appBundle, error) {
	app, err := discovery.NewAppFromString(name)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("invalid image name %q", name), err)
	}
	if _, ok := app.Labels["arch"]; !ok {
		app.Labels["arch"] = runtime.GOARCH
	}
	if _, ok := app.Labels["os"]; !ok {
		app.Labels["os"] = runtime.GOOS
	}
	if err := types.IsValidOSArch(app.Labels, stage0.ValidOSArch); err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("invalid image name %q", name), err)
	}
	bundle := &appBundle{
		App: app,
		Str: name,
	}
	return bundle, nil
}

func (f *Fetcher) fetchSingleImageByName(name string, a *asc) (string, error) {
	app, err := newAppBundle(name)
	if err != nil {
		return "", err
	}
	if h, err := f.maybeCheckStoreForApp(app); h != "" || err != nil {
		return h, err
	}
	if h, err := f.maybeFetchImageFromRemote(app, a); h != "" || err != nil {
		return h, err
	}
	return "", fmt.Errorf("unable to fetch image from image name %q: either image was not found in the store or store was disabled and fetching from remote yielded nothing or it was disabled", name)
}

func (f *Fetcher) maybeCheckStoreForApp(app *appBundle) (string, error) {
	if !f.NoStore {
		key, err := f.getStoreKeyFromApp(app)
		if err == nil {
			log.Printf("using image from local store for image name %s", app.Str)
			return key, nil
		}
		switch err.(type) {
		case store.ACINotFoundError:
			// ignore the "not found" error
		default:
			return "", err
		}
	}
	return "", nil
}

func (f *Fetcher) getStoreKeyFromApp(app *appBundle) (string, error) {
	labels, err := types.LabelsFromMap(app.App.Labels)
	if err != nil {
		return "", errwrap.Wrap(fmt.Errorf("invalid labels in the name %q", app.Str), err)
	}
	key, err := f.S.GetACI(app.App.Name, labels)
	if err != nil {
		switch err.(type) {
		case store.ACINotFoundError:
			return "", err
		default:
			return "", errwrap.Wrap(fmt.Errorf("cannot find image %q", app.Str), err)
		}
	}
	return key, nil
}

func (f *Fetcher) maybeFetchImageFromRemote(app *appBundle, a *asc) (string, error) {
	if !f.StoreOnly {
		nf := &nameFetcher{
			InsecureFlags:      f.InsecureFlags,
			S:                  f.S,
			Ks:                 f.Ks,
			Debug:              f.Debug,
			Headers:            f.Headers,
			TrustKeysFromHTTPS: f.TrustKeysFromHTTPS,
		}
		return nf.GetHash(app.App, a)
	}
	return "", nil
}
