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
	"io/ioutil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	dist "github.com/coreos/rkt/pkg/distribution"
	"github.com/coreos/rkt/pkg/keystore"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/rkt/config"
	rktflag "github.com/coreos/rkt/rkt/flag"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/discovery"
	"github.com/appc/spec/schema"
	"golang.org/x/crypto/openpgp"
)

type imageStringType int

const (
	imageStringName imageStringType = iota // image type to be guessed
	imageStringPath                        // absolute or relative path

	PullPolicyNever  = "never"
	PullPolicyNew    = "new"
	PullPolicyUpdate = "update"
)

// action is a common type for Finder and Fetcher
type action struct {
	// S is an aci store where images will be looked for or stored.
	S *imagestore.Store
	// Ts is an aci tree store.
	Ts *treestore.Store
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

	// PullPolicy controls when to pull images from remote, versus using a copy
	// on the local filesystem, versus checking for updates to local images
	PullPolicy string
	// NoCache tells to ignore transport caching.
	NoCache bool
	// WithDeps tells whether image dependencies should be
	// downloaded too.
	WithDeps bool
}

var (
	log    *rktlog.Logger
	diag   *rktlog.Logger
	stdout *rktlog.Logger
)

func ensureLogger(debug bool) {
	if log == nil || diag == nil || stdout == nil {
		log, diag, stdout = rktlog.NewLogSet("image", debug)
	}
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}
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

// DistFromImageString return the distribution for the given input image string
func DistFromImageString(is string) (dist.Distribution, error) {
	u, err := url.Parse(is)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("failed to parse image url %q", is), err)
	}

	// Convert user friendly image string names to internal distribution URIs
	// file:///full/path/to/aci/file.aci -> archive:aci:file%3A%2F%2F%2Ffull%2Fpath%2Fto%2Faci%2Ffile.aci
	switch u.Scheme {
	case "":
		// no scheme given, hence it is an appc image name or path
		appImageType := guessAppcOrPath(is, []string{schema.ACIExtension})

		switch appImageType {
		case imageStringName:
			app, err := discovery.NewAppFromString(is)
			if err != nil {
				return nil, fmt.Errorf("invalid appc image string %q: %v", is, err)
			}
			return dist.NewAppcFromApp(app), nil
		case imageStringPath:
			absPath, err := filepath.Abs(is)
			if err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("failed to get an absolute path for %q", is), err)
			}
			is = "file://" + absPath

			// given a file:// image string, call this function again to return an ACI distribution
			return DistFromImageString(is)
		default:
			return nil, fmt.Errorf("invalid image string type %q", appImageType)
		}
	case "file", "http", "https":
		// An ACI archive with any transport type (file, http, s3 etc...) and final aci extension
		if filepath.Ext(u.Path) == schema.ACIExtension {
			dist, err := dist.NewACIArchiveFromTransportURL(u)
			if err != nil {
				return nil, fmt.Errorf("archive distribution creation error: %v", err)
			}
			return dist, nil
		}
	case "docker":
		// Accept both docker: and docker:// uri
		dockerStr := is
		if strings.HasPrefix(dockerStr, "docker://") {
			dockerStr = strings.TrimPrefix(dockerStr, "docker://")
		} else if strings.HasPrefix(dockerStr, "docker:") {
			dockerStr = strings.TrimPrefix(dockerStr, "docker:")
		}

		dist, err := dist.NewDockerFromString(dockerStr)
		if err != nil {
			return nil, fmt.Errorf("docker distribution creation error: %v", err)
		}
		return dist, nil
	case dist.Scheme: // cimd
		return dist.Parse(is)
	default:
		// any other scheme is a an appc image name, i.e. "my-app:v1.0"
		app, err := discovery.NewAppFromString(is)
		if err != nil {
			return nil, fmt.Errorf("invalid appc image string %q: %v", is, err)
		}

		return dist.NewAppcFromApp(app), nil
	}

	return nil, fmt.Errorf("invalid image string %q", is)
}

func guessAppcOrPath(is string, extensions []string) imageStringType {
	if filepath.IsAbs(is) {
		return imageStringPath
	}

	// Well, at this point is basically heuristics time. The image
	// parameter can be either a relative path or an image name.

	// First, let's try to stat whatever file the URL would specify. If it
	// exists, that's probably what the user wanted.
	f, err := os.Stat(is)
	if err == nil && f.Mode().IsRegular() {
		return imageStringPath
	}

	// Second, let's check if there is a colon in the image
	// parameter. Colon often serves as a paths separator (like in
	// the PATH environment variable), so if it exists, then it is
	// highly unlikely that the image parameter is a path. Colon
	// in this context is often used for specifying a version of
	// an image, like in "example.com/reduce-worker:1.0.0".
	if strings.ContainsRune(is, ':') {
		return imageStringName
	}

	// Third, let's check if there is a dot followed by a slash
	// (./) - if so, it is likely that the image parameter is path
	// like ./aci-in-this-dir or ../aci-in-parent-dir
	if strings.Contains(is, "./") {
		return imageStringPath
	}

	// Fourth, let's check if the image parameter has an .aci
	// extension. If so, likely a path like "stage1-coreos.aci".
	for _, e := range extensions {
		if filepath.Ext(is) == e {
			return imageStringPath
		}
	}

	// At this point, if the image parameter is something like
	// "coreos.com/rkt/stage1-coreos" and you have a directory
	// tree "coreos.com/rkt" in the current working directory and
	// you meant the image parameter to point to the file
	// "stage1-coreos" in this directory tree, then you better be
	// off prepending the parameter with "./", because I'm gonna
	// treat this as an image name otherwise.
	return imageStringName
}

func eTag(rem *imagestore.Remote) string {
	if rem != nil {
		return rem.ETag
	}
	return ""
}

func maybeUseCached(rem *imagestore.Remote, cd *cacheData) string {
	if rem == nil || cd == nil {
		return ""
	}
	if cd.UseCached {
		return rem.BlobKey
	}
	return ""
}

func remoteForURL(s *imagestore.Store, u *url.URL) (*imagestore.Remote, error) {
	urlStr := u.String()
	rem, err := s.GetRemote(urlStr)
	if err != nil {
		if err == imagestore.ErrRemoteNotFound {
			return nil, nil
		}

		return nil, errwrap.Wrap(fmt.Errorf("failed to fetch remote for URL %q", urlStr), err)
	}

	return rem, nil
}
