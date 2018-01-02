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
	"io/ioutil"
	"net/url"
	"os"
	"path"
	"strings"
	"time"

	"github.com/coreos/rkt/rkt/config"
	rktflag "github.com/coreos/rkt/rkt/flag"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/hashicorp/errwrap"

	docker2aci "github.com/appc/docker2aci/lib"
	d2acommon "github.com/appc/docker2aci/lib/common"
)

// dockerFetcher is used to fetch images from docker:// URLs. It uses
// a docker2aci library to perform this task.
type dockerFetcher struct {
	// TODO(krnowak): Fix the docs when we support docker image
	// verification. Will that ever happen?
	// InsecureFlags tells which insecure functionality should
	// be enabled. No image verification must be true for now.
	InsecureFlags *rktflag.SecFlags
	DockerAuth    map[string]config.BasicCredentials
	S             *imagestore.Store
	Debug         bool
}

// Hash uses docker2aci to download the image and convert it to
// ACI, then stores it in the store and returns the hash.
func (f *dockerFetcher) Hash(u *url.URL) (string, error) {
	ensureLogger(f.Debug)
	dockerURL, err := d2acommon.ParseDockerURL(path.Join(u.Host, u.Path))
	if err != nil {
		return "", fmt.Errorf(`invalid docker URL %q; expected syntax is "docker://[REGISTRY_HOST[:REGISTRY_PORT]/]IMAGE_NAME[:TAG]"`, u)
	}
	latest := dockerURL.Tag == "latest"
	return f.fetchImageFrom(u, latest)
}

func (f *dockerFetcher) fetchImageFrom(u *url.URL, latest bool) (string, error) {
	if !f.InsecureFlags.SkipImageCheck() {
		return "", fmt.Errorf("signature verification for docker images is not supported (try --insecure-options=image)")
	}

	diag.Printf("fetching image from %s", u.String())

	aciFile, err := f.fetch(u)
	if err != nil {
		return "", err
	}
	// At this point, the ACI file is removed, but it is kept
	// alive, because we have an fd to it opened.
	defer aciFile.Close()

	key, err := f.S.WriteACI(aciFile, imagestore.ACIFetchInfo{
		Latest: latest,
	})
	if err != nil {
		return "", err
	}

	// docker images don't have signature URL
	newRem := imagestore.NewRemote(u.String(), "")
	newRem.BlobKey = key
	newRem.DownloadTime = time.Now()
	err = f.S.WriteRemote(newRem)
	if err != nil {
		return "", err
	}

	return key, nil
}

func (f *dockerFetcher) fetch(u *url.URL) (*os.File, error) {
	tmpDir, err := f.getTmpDir()
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	registryURL := strings.TrimPrefix(u.String(), "docker://")
	user, password := f.getCreds(registryURL)
	config := docker2aci.RemoteConfig{
		Username: user,
		Password: password,
		Insecure: d2acommon.InsecureConfig{
			SkipVerify: f.InsecureFlags.SkipTLSCheck(),
			AllowHTTP:  f.InsecureFlags.AllowHTTP(),
		},
		CommonConfig: docker2aci.CommonConfig{
			Squash:      true,
			OutputDir:   tmpDir,
			TmpDir:      tmpDir,
			Compression: d2acommon.NoCompression,
		},
	}
	acis, err := docker2aci.ConvertRemoteRepo(registryURL, config)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error converting docker image to ACI"), err)
	}

	aciFile, err := os.Open(acis[0])
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error opening squashed ACI file"), err)
	}

	return aciFile, nil
}

func (f *dockerFetcher) getTmpDir() (string, error) {
	storeTmpDir, err := f.S.TmpDir()
	if err != nil {
		return "", errwrap.Wrap(errors.New("error creating temporary dir for docker to ACI conversion"), err)
	}
	return ioutil.TempDir(storeTmpDir, "docker2aci-")
}

func (f *dockerFetcher) getCreds(registryURL string) (string, string) {
	indexName := docker2aci.GetIndexName(registryURL)
	if creds, ok := f.DockerAuth[indexName]; ok {
		return creds.User, creds.Password
	}
	return "", ""
}
