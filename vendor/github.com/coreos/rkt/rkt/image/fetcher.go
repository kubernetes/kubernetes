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
	"errors"
	"fmt"
	"net/url"
	"os"
	"runtime"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/apps"
	dist "github.com/coreos/rkt/pkg/distribution"
	"github.com/coreos/rkt/stage0"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/discovery"
	"github.com/appc/spec/schema/types"
)

// distBundle contains the distribution and the original image string
// (primarily used for logging)
type distBundle struct {
	dist  dist.Distribution
	image string
}

// Fetcher will try to fetch images into the store.
type Fetcher action

// FetchImages uses FetchImage to attain a list of image hashes
func (f *Fetcher) FetchImages(al *apps.Apps) error {
	return al.Walk(func(app *apps.App) error {
		d, err := DistFromImageString(app.Image)
		if err != nil {
			return err
		}
		h, err := f.FetchImage(d, app.Image, app.Asc)
		if err != nil {
			return err
		}
		app.ImageID = *h
		return nil
	})
}

// FetchImage will take an image as either a path, a URL or a name
// string and import it into the store if found. If ascPath is not "",
// it must exist as a local file and will be used as the signature
// file for verification, unless verification is disabled. If
// f.WithDeps is true also image dependencies are fetched.
func (f *Fetcher) FetchImage(d dist.Distribution, image, ascPath string) (*types.Hash, error) {
	ensureLogger(f.Debug)
	db := &distBundle{
		dist:  d,
		image: image,
	}
	a := f.getAsc(ascPath)
	hash, err := f.fetchSingleImage(db, a)
	if err != nil {
		return nil, err
	}
	if f.WithDeps {
		err = f.fetchImageDeps(hash)
		if err != nil {
			return nil, err
		}
	}

	// we need to be able to do a chroot and access to the tree store
	// directories, we need to
	// 1) check if the system supports OverlayFS
	// 2) check if we're root
	if common.SupportsOverlay() == nil && os.Geteuid() == 0 {
		if _, _, err := f.Ts.Render(hash, false); err != nil {
			return nil, errwrap.Wrap(errors.New("error rendering tree store"), err)
		}
	}
	h, err := types.NewHash(hash)
	if err != nil {
		// should never happen
		log.PanicE("invalid hash", err)
	}
	return h, nil
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
	seen := map[string]dist.Distribution{}
	f.addImageDeps(hash, imgsl, seen)
	for el := imgsl.Front(); el != nil; el = el.Next() {
		a := &asc{}
		d := el.Value.(*dist.Appc)
		str := d.String()
		db := &distBundle{
			dist:  d,
			image: str,
		}
		hash, err := f.fetchSingleImage(db, a)
		if err != nil {
			return err
		}
		f.addImageDeps(hash, imgsl, seen)
	}
	return nil
}

func (f *Fetcher) addImageDeps(hash string, imgsl *list.List, seen map[string]dist.Distribution) error {
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
		d := dist.NewAppcFromApp(app)
		// To really catch already seen deps the saved string must be a
		// reproducible string keeping the labels order
		for _, sd := range seen {
			if d.Equals(sd) {
				continue
			}
		}
		imgsl.PushBack(d)
		seen[d.CIMD().String()] = d
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

func (f *Fetcher) fetchSingleImage(db *distBundle, a *asc) (string, error) {
	switch v := db.dist.(type) {
	case *dist.ACIArchive:
		return f.fetchACIArchive(db, a)
	case *dist.Appc:
		return f.fetchSingleImageByName(db, a)
	case *dist.Docker:
		return f.fetchSingleImageByDockerURL(v)
	default:
		return "", fmt.Errorf("unknown distribution type %T", v)
	}
}

func (f *Fetcher) fetchACIArchive(db *distBundle, a *asc) (string, error) {
	u := db.dist.(*dist.ACIArchive).TransportURL()

	switch u.Scheme {
	case "http", "https":
		return f.fetchSingleImageByHTTPURL(u, a)
	case "file":
		return f.fetchSingleImageByPath(u.Path, a)
	case "":
		return "", fmt.Errorf("expected image URL %q to contain a scheme", u.String())
	default:
		return "", fmt.Errorf("an unsupported URL scheme %q - the only URL schemes supported by rkt for an archive are http, https and file", u.Scheme)
	}
}

func (f *Fetcher) fetchSingleImageByHTTPURL(u *url.URL, a *asc) (string, error) {
	rem, err := remoteForURL(f.S, u)
	if err != nil {
		return "", err
	}
	if h := f.maybeCheckRemoteFromStore(rem); h != "" {
		return h, nil
	}
	if h, err := f.maybeFetchHTTPURLFromRemote(rem, u, a); h != "" || err != nil {
		return h, err
	}
	return "", fmt.Errorf("unable to fetch image from URL %q: either image was not found in the store or store was disabled and fetching from remote yielded nothing or it was disabled", u.String())
}

func (f *Fetcher) fetchSingleImageByDockerURL(d *dist.Docker) (string, error) {
	ds := d.ReferenceURL()
	// Convert to the docker2aci URL format
	urlStr := "docker://" + ds
	u, err := url.Parse(urlStr)
	if err != nil {
		return "", err
	}

	rem, err := remoteForURL(f.S, u)
	if err != nil {
		return "", err
	}
	if h := f.maybeCheckRemoteFromStore(rem); h != "" {
		return h, nil
	}
	if h, err := f.maybeFetchDockerURLFromRemote(u); h != "" || err != nil {
		return h, err
	}
	return "", fmt.Errorf("unable to fetch docker image from URL %q: either image was not found in the store or store was disabled and fetching from remote yielded nothing or it was disabled", u.String())
}

func (f *Fetcher) maybeCheckRemoteFromStore(rem *imagestore.Remote) string {
	if f.PullPolicy == PullPolicyUpdate || rem == nil {
		return ""
	}
	diag.Printf("using image from local store for url %s", rem.ACIURL)
	return rem.BlobKey
}

func (f *Fetcher) maybeFetchHTTPURLFromRemote(rem *imagestore.Remote, u *url.URL, a *asc) (string, error) {
	if f.PullPolicy != PullPolicyNever {
		diag.Printf("remote fetching from URL %q", u.String())
		hf := &httpFetcher{
			InsecureFlags: f.InsecureFlags,
			S:             f.S,
			Ks:            f.Ks,
			Rem:           rem,
			Debug:         f.Debug,
			Headers:       f.Headers,
		}
		return hf.Hash(u, a)
	}
	return "", nil
}

func (f *Fetcher) maybeFetchDockerURLFromRemote(u *url.URL) (string, error) {
	if f.PullPolicy != PullPolicyNever {
		diag.Printf("remote fetching from URL %q", u.String())
		df := &dockerFetcher{
			InsecureFlags: f.InsecureFlags,
			DockerAuth:    f.DockerAuth,
			S:             f.S,
			Debug:         f.Debug,
		}
		return df.Hash(u)
	}
	return "", nil
}

func (f *Fetcher) fetchSingleImageByPath(path string, a *asc) (string, error) {
	diag.Printf("using image from file %s", path)
	ff := &fileFetcher{
		InsecureFlags: f.InsecureFlags,
		S:             f.S,
		Ks:            f.Ks,
		Debug:         f.Debug,
	}
	return ff.Hash(path, a)
}

// TODO(sgotti) I'm not sure setting default os and arch also for image
// dependencies is correct since it may break noarch dependencies. Reference:
// https://github.com/coreos/rkt/pull/2317
func (db *distBundle) setAppDefaults() error {
	app := db.dist.(*dist.Appc).App()
	if _, ok := app.Labels["arch"]; !ok {
		app.Labels["arch"] = runtime.GOARCH
	}
	if _, ok := app.Labels["os"]; !ok {
		app.Labels["os"] = runtime.GOOS
	}
	if err := types.IsValidOSArch(app.Labels, stage0.ValidOSArch); err != nil {
		return errwrap.Wrap(fmt.Errorf("invalid Appc distribution %q", db.image), err)
	}
	db.dist = dist.NewAppcFromApp(app)
	return nil
}

func (f *Fetcher) fetchSingleImageByName(db *distBundle, a *asc) (string, error) {
	if err := db.setAppDefaults(); err != nil {
		return "", err
	}
	if h, err := f.maybeCheckStoreForApp(db); h != "" || err != nil {
		return h, err
	}
	if h, err := f.maybeFetchImageFromRemote(db, a); h != "" || err != nil {
		return h, err
	}
	return "", fmt.Errorf("unable to fetch image from image name %q: either image was not found in the store or store was disabled and fetching from remote yielded nothing or it was disabled", db.image)
}

func (f *Fetcher) maybeCheckStoreForApp(db *distBundle) (string, error) {
	if f.PullPolicy != PullPolicyUpdate {
		key, err := f.getStoreKeyFromApp(db)
		if err == nil {
			diag.Printf("using image from local store for image name %s", db.image)
			return key, nil
		}
		switch err.(type) {
		case imagestore.ACINotFoundError:
			// ignore the "not found" error
		default:
			return "", err
		}
	}
	return "", nil
}

func (f *Fetcher) getStoreKeyFromApp(db *distBundle) (string, error) {
	app := db.dist.(*dist.Appc).App()
	labels, err := types.LabelsFromMap(app.Labels)
	if err != nil {
		return "", errwrap.Wrap(fmt.Errorf("invalid labels in the name %q", db.image), err)
	}
	key, err := f.S.GetACI(app.Name, labels)
	if err != nil {
		switch err.(type) {
		case imagestore.ACINotFoundError:
			return "", err
		default:
			return "", errwrap.Wrap(fmt.Errorf("cannot find image %q", db.image), err)
		}
	}
	return key, nil
}

func (f *Fetcher) maybeFetchImageFromRemote(db *distBundle, a *asc) (string, error) {
	if f.PullPolicy != PullPolicyNever {
		app := db.dist.(*dist.Appc).App()
		nf := &nameFetcher{
			InsecureFlags:      f.InsecureFlags,
			S:                  f.S,
			Ks:                 f.Ks,
			NoCache:            f.NoCache,
			Debug:              f.Debug,
			Headers:            f.Headers,
			TrustKeysFromHTTPS: f.TrustKeysFromHTTPS,
		}
		return nf.Hash(app, a)
	}
	return "", nil
}
