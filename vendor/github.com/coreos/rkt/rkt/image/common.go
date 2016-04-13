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
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/coreos/rkt/common/apps"
	"github.com/coreos/rkt/pkg/keystore"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/rkt/config"
	rktflag "github.com/coreos/rkt/rkt/flag"
	"github.com/coreos/rkt/store"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"golang.org/x/crypto/openpgp"
)

// action is a common type for Finder and Fetcher
type action struct {
	// S is a store where images will be looked for or stored.
	S *store.Store
	// Ks is a keystore used for verification of the image
	Ks *keystore.Keystore
	// Headers is a map of headers which might be used for
	// downloading via https protocol.
	Headers map[string]config.Headerer
	// DockerAuth is used for authenticating when fetching docker
	// images.
	DockerAuth map[string]config.BasicCredentials
	// InsecureFlags is a set of flags for enabling some insecure
	// functionality. For now it is mostly skipping image
	// signature verification and TLS certificate verification.
	InsecureFlags *rktflag.SecFlags
	// Debug tells whether additional debug messages should be
	// printed.
	Debug bool
	// TrustKeysFromHTTPS tells whether discovered keys downloaded
	// via the https protocol can be trusted
	TrustKeysFromHTTPS bool

	// StoreOnly tells whether to avoid getting images from a
	// local filesystem or a remote location.
	StoreOnly bool
	// NoStore tells whether to avoid getting images from the
	// store. Note that the store can be still used as a cache.
	NoStore bool
	// NoCache tells to avoid getting images from the store
	// completely.
	NoCache bool
	// WithDeps tells whether image dependencies should be
	// downloaded too.
	WithDeps bool
}

var (
	log    *rktlog.Logger
	stdout *rktlog.Logger = rktlog.New(os.Stdout, "", false)
)

func ensureLogger(debug bool) {
	if log == nil {
		log = rktlog.New(os.Stderr, "image", debug)
	}
}

// isReallyNil makes sure that the passed value is really really
// nil. So it returns true if value is plain nil or if it is e.g. an
// interface with non-nil type but nil-value (which normally is
// different from nil itself).
func isReallyNil(iface interface{}) bool {
	// this catches the cases when you pass non-interface nil
	// directly, like:
	//
	// isReallyNil(nil)
	// var m map[string]string
	// isReallyNil(m)
	if iface == nil {
		return true
	}
	// get a reflect value
	v := reflect.ValueOf(iface)
	// only channels, functions, interfaces, maps, pointers and
	// slices are nillable
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		// this catches the cases when you pass some interface
		// with nil value, like:
		//
		// var v io.Closer = func(){var f *os.File; return f}()
		// isReallyNil(v)
		return v.IsNil()
	}
	return false
}

// useCached checks if downloadTime plus maxAge is before/after the current time.
// return true if the cached image should be used, false otherwise.
func useCached(downloadTime time.Time, maxAge int) bool {
	freshnessLifetime := int(time.Now().Sub(downloadTime).Seconds())
	if maxAge > 0 && freshnessLifetime < maxAge {
		return true
	}
	return false
}

// ascURLFromImgURL creates a URL to a signature file from passed URL
// to an image.
func ascURLFromImgURL(u *url.URL) *url.URL {
	copy := *u
	copy.Path = ascPathFromImgPath(copy.Path)
	return &copy
}

// ascPathFromImgPath creates a path to a signature file from passed
// path to an image.
func ascPathFromImgPath(path string) string {
	return fmt.Sprintf("%s.aci.asc", strings.TrimSuffix(path, ".aci"))
}

// printIdentities prints a message that signature was verified.
func printIdentities(entity *openpgp.Entity) {
	lines := []string{"signature verified:"}
	for _, v := range entity.Identities {
		lines = append(lines, fmt.Sprintf("  %s", v.Name))
	}
	log.Print(strings.Join(lines, "\n"))
}

func guessImageType(image string) apps.AppImageType {
	if _, err := types.NewHash(image); err == nil {
		return apps.AppImageHash
	}
	if u, err := url.Parse(image); err == nil && u.Scheme != "" {
		return apps.AppImageURL
	}
	if filepath.IsAbs(image) {
		return apps.AppImagePath
	}

	// Well, at this point is basically heuristics time. The image
	// parameter can be either a relative path or an image name.

	// First, let's check if there is a colon in the image
	// parameter. Colon often serves as a paths separator (like in
	// the PATH environment variable), so if it exists, then it is
	// highly unlikely that the image parameter is a path. Colon
	// in this context is often used for specifying a version of
	// an image, like in "example.com/reduce-worker:1.0.0".
	if strings.ContainsRune(image, ':') {
		return apps.AppImageName
	}

	// Second, let's check if there is a dot followed by a slash
	// (./) - if so, it is likely that the image parameter is path
	// like ./aci-in-this-dir or ../aci-in-parent-dir
	if strings.Contains(image, "./") {
		return apps.AppImagePath
	}

	// Third, let's check if the image parameter has an .aci
	// extension. If so, likely a path like "stage1-coreos.aci".
	if filepath.Ext(image) == schema.ACIExtension {
		return apps.AppImagePath
	}

	// At this point, if the image parameter is something like
	// "coreos.com/rkt/stage1-coreos" and you have a directory
	// tree "coreos.com/rkt" in the current working directory and
	// you meant the image parameter to point to the file
	// "stage1-coreos" in this directory tree, then you better be
	// off prepending the parameter with "./", because I'm gonna
	// treat this as an image name otherwise.
	return apps.AppImageName
}
