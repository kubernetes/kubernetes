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
	"os"
	"path/filepath"

	"github.com/coreos/rkt/pkg/keystore"
	rktflag "github.com/coreos/rkt/rkt/flag"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/hashicorp/errwrap"
)

// fileFetcher is used to fetch files from a local filesystem
type fileFetcher struct {
	InsecureFlags *rktflag.SecFlags
	S             *imagestore.Store
	Ks            *keystore.Keystore
	Debug         bool
}

// Hash opens a file, optionally verifies it against passed asc,
// stores it in the store and returns the hash.
func (f *fileFetcher) Hash(aciPath string, a *asc) (string, error) {
	ensureLogger(f.Debug)
	absPath, err := filepath.Abs(aciPath)
	if err != nil {
		return "", errwrap.Wrap(fmt.Errorf("failed to get an absolute path for %q", aciPath), err)
	}
	aciPath = absPath

	aciFile, err := f.getFile(aciPath, a)
	if err != nil {
		return "", err
	}
	defer aciFile.Close()

	key, err := f.S.WriteACI(aciFile, imagestore.ACIFetchInfo{
		Latest: false,
	})
	if err != nil {
		return "", err
	}

	return key, nil
}

func (f *fileFetcher) getFile(aciPath string, a *asc) (*os.File, error) {
	if f.InsecureFlags.SkipImageCheck() && f.Ks != nil {
		log.Printf("warning: image signature verification has been disabled")
	}
	if f.InsecureFlags.SkipImageCheck() || f.Ks == nil {
		aciFile, err := os.Open(aciPath)
		if err != nil {
			return nil, errwrap.Wrap(errors.New("error opening ACI file"), err)
		}
		return aciFile, nil
	}
	aciFile, err := f.getVerifiedFile(aciPath, a)
	if err != nil {
		return nil, err
	}
	return aciFile, nil
}

// fetch opens and verifies the ACI.
func (f *fileFetcher) getVerifiedFile(aciPath string, a *asc) (*os.File, error) {
	var aciFile *os.File // closed on error
	var errClose error   // error signaling to close aciFile

	f.maybeOverrideAsc(aciPath, a)
	ascFile, err := a.Get()
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error opening signature file"), err)
	}
	defer ascFile.Close()

	aciFile, err = os.Open(aciPath)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error opening ACI file"), err)
	}

	defer func() {
		if errClose != nil {
			aciFile.Close()
		}
	}()

	validator, errClose := newValidator(aciFile)
	if errClose != nil {
		return nil, errClose
	}

	entity, errClose := validator.ValidateWithSignature(f.Ks, ascFile)
	if errClose != nil {
		return nil, errwrap.Wrap(fmt.Errorf("image %q verification failed", validator.ImageName()), errClose)
	}
	printIdentities(entity)

	return aciFile, nil
}

func (f *fileFetcher) maybeOverrideAsc(aciPath string, a *asc) {
	if a.Fetcher != nil {
		return
	}
	a.Location = ascPathFromImgPath(aciPath)
	a.Fetcher = &localAscFetcher{}
}
