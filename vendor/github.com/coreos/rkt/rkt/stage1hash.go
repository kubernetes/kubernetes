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

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common/apps"
	"github.com/coreos/rkt/rkt/config"
	"github.com/coreos/rkt/rkt/image"
	"github.com/coreos/rkt/store"
	"github.com/hashicorp/errwrap"
	"github.com/spf13/pflag"
)

// stage1ImageLocationKind describes the stage1 image location
type stage1ImageLocationKind int

const (
	// location unset, it is not a valid kind to be used
	stage1ImageLocationUnset stage1ImageLocationKind = iota
	// a URL with a scheme
	stage1ImageLocationURL
	// an absolute or a relative path
	stage1ImageLocationPath
	// an image name
	stage1ImageLocationName
	// an image hash
	stage1ImageLocationHash
	// an image in the default dir
	stage1ImageLocationFromDir
)

// stage1FlagData is used for creating the flags for each valid location kind
type stage1FlagData struct {
	kind stage1ImageLocationKind
	flag string
	name string
	help string
}

// stage1ImageLocation is used to store the user's choice of stage1 image via flags
type stage1ImageLocation struct {
	kind     stage1ImageLocationKind
	location string
}

// stage1ImageLocationFlag is an implementation of a pflag.Value
// interface, which handles all the valid location kinds
type stage1ImageLocationFlag struct {
	loc  *stage1ImageLocation
	kind stage1ImageLocationKind
}

func (f *stage1ImageLocationFlag) Set(location string) error {
	if f.loc.kind != stage1ImageLocationUnset {
		wanted := stage1FlagsData[f.kind]
		current := stage1FlagsData[f.loc.kind]
		if f.loc.kind == f.kind {
			return fmt.Errorf("--%s already used", current.flag)
		}
		return fmt.Errorf("flags --%s and --%s are mutually exclusive",
			wanted.flag, current.flag)
	}
	f.loc.kind = f.kind
	f.loc.location = location
	return nil
}

func (f *stage1ImageLocationFlag) String() string {
	return f.loc.location
}

func (f *stage1ImageLocationFlag) Type() string {
	return stage1FlagsData[f.kind].name
}

var (
	// defaults defined by configure, set by linker
	// default stage1 image name
	// (e.g. coreos.com/rkt/stage1-coreos)
	buildDefaultStage1Name string
	// default stage1 image version (e.g. 0.15.0)
	buildDefaultStage1Version string
	// an absolute path or a URL to the default stage1 image file
	buildDefaultStage1ImageLoc string
	// filename of the default stage1 image file in the default
	// stage1 images directory
	buildDefaultStage1ImageInRktDir string
	// an absolute path to the stage1 images directory
	buildDefaultStage1ImagesDir string

	// this holds necessary data to generate the --stage1-* flags
	// for each location kind
	stage1FlagsData = map[stage1ImageLocationKind]*stage1FlagData{
		stage1ImageLocationURL: &stage1FlagData{
			kind: stage1ImageLocationURL,
			flag: "stage1-url",
			name: "stage1URL",
			help: "a URL to an image to use as stage1",
		},

		stage1ImageLocationPath: &stage1FlagData{
			kind: stage1ImageLocationPath,
			flag: "stage1-path",
			name: "stage1Path",
			help: "an absolute or a relative path to an image to use as stage1",
		},

		stage1ImageLocationName: &stage1FlagData{
			kind: stage1ImageLocationName,
			flag: "stage1-name",
			name: "stage1Name",
			help: "a name of an image to use as stage1",
		},

		stage1ImageLocationHash: &stage1FlagData{
			kind: stage1ImageLocationHash,
			flag: "stage1-hash",
			name: "stage1Hash",
			help: "a hash of an image to use as stage1",
		},

		stage1ImageLocationFromDir: &stage1FlagData{
			kind: stage1ImageLocationFromDir,
			flag: "stage1-from-dir",
			name: "stage1FromDir",
			help: "a filename of an image in stage1 images directory to use as stage1",
		},
	}
	// location to stage1 image overridden by one of --stage1-*
	// flags
	overriddenStage1Location = stage1ImageLocation{
		kind:     stage1ImageLocationUnset,
		location: "",
	}
)

// addStage1ImageFlags adds flags for specifying custom stage1 image
func addStage1ImageFlags(flags *pflag.FlagSet) {
	for _, data := range stage1FlagsData {
		wrapper := &stage1ImageLocationFlag{
			loc:  &overriddenStage1Location,
			kind: data.kind,
		}
		flags.Var(wrapper, data.flag, data.help)
	}
}

// getStage1Hash will try to get the hash of stage1 to use.
//
// Before getting inside this rats nest, let's try to write up the
// expected behaviour.
//
// If the user passed --stage1-url, --stage1-path, --stage1-name,
// --stage1-hash, or --stage1-from-dir then we take what was passed
// and try to load it. If it failed, we bail out. No second chances
// and whatnot. The details about how each location type should be
// handled are below.
//
// If the user passed none of the above flags, we try to get the name,
// the version and the location from the configuration. The name and
// the version must be defined in pair, that is - either both of them
// are defined in configuration or none. Values from the configuration
// override the values taken from the configure script. We search for
// an image with the default name and version in the store. If it is
// there, then woo, we are done. Otherwise we get the location and try
// to load it. Depending on location type, we bail out immediately or
// get a second chance.
//
// Details about the handling of different location types follow.
//
// If location is a URL, we do no discovery, just try to fetch it
// directly into the store instead.
//
// If location is a path, we do no discovery, just try to fetch it
// directly into the store instead. If the file is not found and we
// have a second chance, we try to fetch the file in the same
// directory as the rkt binary itself into the store.
//
// If location is a name, we do the discovery, fetch the discovered
// image into the store.
//
// If location is an image hash then we make sure that it exists in
// the store.
func getStage1Hash(s *store.Store, c *config.Config) (*types.Hash, error) {
	imgDir := getStage1ImagesDirectory(c)
	if overriddenStage1Location.kind != stage1ImageLocationUnset {
		// we passed a --stage-{url,path,name,hash,from-dir} flag
		return getStage1HashFromFlag(s, overriddenStage1Location, imgDir)
	}

	imgRef, imgLoc, imgFileName := getStage1DataFromConfig(c)
	return getConfiguredStage1Hash(s, imgRef, imgLoc, imgFileName)
}

func getStage1ImagesDirectory(c *config.Config) string {
	if c.Paths.Stage1ImagesDir != "" {
		return c.Paths.Stage1ImagesDir
	}
	return buildDefaultStage1ImagesDir
}

func getStage1HashFromFlag(s *store.Store, loc stage1ImageLocation, dir string) (*types.Hash, error) {
	imgType := apps.AppImageGuess
	switch loc.kind {
	case stage1ImageLocationURL:
		imgType = apps.AppImageURL
	case stage1ImageLocationPath:
		imgType = apps.AppImagePath
	case stage1ImageLocationName:
		imgType = apps.AppImageName
	case stage1ImageLocationHash:
		imgType = apps.AppImageHash
	case stage1ImageLocationFromDir:
		loc.location = filepath.Join(dir, loc.location)
		imgType = apps.AppImagePath
	}

	fn := getStage1Finder(s)
	return fn.FindImage(loc.location, "", imgType)
}

func getStage1DataFromConfig(c *config.Config) (string, string, string) {
	imgName := c.Stage1.Name
	imgVersion := c.Stage1.Version
	// if the name in the configuration is empty, then the version
	// is empty too, but let's better be safe now then sorry later
	// - if either one is empty we take build defaults for both
	if imgName == "" || imgVersion == "" {
		imgName = buildDefaultStage1Name
		imgVersion = buildDefaultStage1Version
	}
	imgRef := fmt.Sprintf("%s:%s", imgName, imgVersion)

	imgLoc := c.Stage1.Location
	imgFileName := getFileNameFromLocation(imgLoc)
	if imgLoc == "" {
		imgLoc = buildDefaultStage1ImageLoc
		imgFileName = buildDefaultStage1ImageInRktDir
	}

	return imgRef, imgLoc, imgFileName
}

func getFileNameFromLocation(imgLoc string) string {
	if !filepath.IsAbs(imgLoc) {
		return ""
	}
	return filepath.Base(imgLoc)
}

func getConfiguredStage1Hash(s *store.Store, imgRef, imgLoc, imgFileName string) (*types.Hash, error) {
	fn := getStage1Finder(s)
	if !strings.HasSuffix(imgRef, "-dirty") {
		fn.StoreOnly = true
		if hash, err := fn.FindImage(imgRef, "", apps.AppImageName); err == nil {
			return hash, nil
		}
		fn.StoreOnly = false
	}
	if imgLoc == "" && imgFileName == "" {
		return nil, fmt.Errorf("neither the location of the default stage1 image nor its filename are set, use --stage1-{path,url,name,hash,from-dir} flag")
	}
	// If imgLoc is not an absolute path, then it is a URL
	imgLocIsURL := imgLoc != "" && !filepath.IsAbs(imgLoc)
	if imgLocIsURL {
		return fn.FindImage(imgLoc, "", apps.AppImageURL)
	}
	return getStage1HashFromPath(fn, imgLoc, imgFileName)
}

func getStage1Finder(s *store.Store) *image.Finder {
	return &image.Finder{
		S:                  s,
		InsecureFlags:      globalFlags.InsecureFlags,
		TrustKeysFromHTTPS: globalFlags.TrustKeysFromHTTPS,

		StoreOnly: false,
		NoStore:   false,
		WithDeps:  false,
	}
}

func getStage1HashFromPath(fn *image.Finder, imgLoc, imgFileName string) (*types.Hash, error) {
	var fetchErr error
	var fallbackErr error
	if imgLoc != "" {
		hash, err := fn.FindImage(imgLoc, "", apps.AppImagePath)
		if err == nil {
			return hash, nil
		}
		fetchErr = err
	}
	if imgFileName != "" {
		exePath, err := os.Readlink("/proc/self/exe")
		if err != nil {
			fallbackErr = err
		} else {
			rktDir := filepath.Dir(exePath)
			imgPath := filepath.Join(rktDir, imgFileName)
			hash, err := fn.FindImage(imgPath, "", apps.AppImagePath)
			if err == nil {
				return hash, nil
			}
			fallbackErr = err
		}
	}
	return nil, mergeStage1Errors(fetchErr, fallbackErr)
}

func mergeStage1Errors(fetchErr, fallbackErr error) error {
	if fetchErr != nil && fallbackErr != nil {
		innerErr := errwrap.Wrap(fallbackErr, fetchErr)
		return errwrap.Wrap(errors.New("failed to fetch stage1 image and failed to fall back to stage1 image in the rkt directory"), innerErr)
	} else if fetchErr != nil {
		return errwrap.Wrap(errors.New("failed to fetch stage1 image"), fetchErr)
	}
	return errwrap.Wrap(errors.New("failed to fall back to stage1 image in rkt directory (default stage1 image location is not specified)"), fallbackErr)
}
