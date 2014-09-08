/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package kube_updown provides functions for deploying a kubernetes cluster
// on Google Compute Engine
package kube_updown

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unsafe"

	compute "code.google.com/p/google-api-go-client/compute/v1"
	"code.google.com/p/google-api-go-client/googleapi"
	storage "code.google.com/p/google-api-go-client/storage/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	"github.com/golang/glog"
)

// #cgo LDFLAGS: -lcrypt
// #define _GNU_SOURCE
// #include <crypt.h>
// #include <stdlib.h>
import "C"

const (
	instancePrefix = "kubernetes"
	MasterName     = instancePrefix + "-master"
	MinionPrefix   = instancePrefix + "-minion"
	masterTag      = MasterName
	minionTag      = MinionPrefix
	masterSize     = "g1-small"
	minionSize     = "g1-small"
	image          = "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/backports-debian-7-wheezy-v20140814"
	minionIPs      = "10.244.%v.0/24"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// CreateMasterFirewall creates a firewall rule to allow https to the kube-master from anywhere.
func CreateMasterFirewall(gce *gce_cloud.GCECloud) (*gce_cloud.GCEOp, error) {
	return gce.CreateFirewall(MasterName+"-https", "0.0.0.0/0", masterTag, "tcp:443")
}

// CreateMinionFirewall creates a firewall rule to allow communication on the internal IPs.
func CreateMinionFirewall(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v-all", MinionPrefix, i)
	ipRange := fmt.Sprintf(minionIPs, i)
	return gce.CreateFirewall(name, ipRange, minionTag, "tcp,udp,icmp,esp,ah,sctp")
}

// CreateMaster spins up the kube-master instance.
func CreateMaster(gce *gce_cloud.GCECloud, project string) (*gce_cloud.GCEOp, error) {
	release, err := getReleaseName(gce, project)
	if err != nil {
		return nil, err
	}
	masterStartup := masterStartupScript(project, release)
	scopes := []string{
		compute.DevstorageRead_onlyScope,
		compute.ComputeScope,
	}
	return gce.CreateInstance(MasterName, masterSize, image, masterTag, string(masterStartup), false, scopes)
}

// CreateMinion spins up the specified minion instance.
func CreateMinion(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v", MinionPrefix, i)
	minionStartup := minionStartupScript(i)
	return gce.CreateInstance(name, minionSize, image, minionTag, string(minionStartup), true, []string{})
}

// CreateMinionRoute adds the minion routing rule to allow forwarding to the pods
func CreateMinionRoute(gce *gce_cloud.GCECloud, project, zone string, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v", MinionPrefix, i)
	nextHop := fmt.Sprintf("%v/zones/%v/instances/%v", gce_cloud.FullQualProj(project), zone, name)
	ipRange := fmt.Sprintf("10.244.%v.0/24", i)
	return gce.CreateRoute(name, nextHop, ipRange)
}

// GetCredentials gets the credentials stored in the .kubernetes_auth file (or create and store new credentials).
func GetCredentials() (string, string) {
	usr, err := user.Current()
	var homeDir string
	if err == nil {
		homeDir = usr.HomeDir
	} else {
		homeDir = "."
	}

	fileName := filepath.Join(homeDir, ".kubernetes_auth")
	if _, err := os.Stat(fileName); os.IsNotExist(err) {
		usr := "admin"
		passwd := randAlphaNumString(16, false)
		a := authFile{
			User:     usr,
			Password: passwd,
		}
		auth, err := json.Marshal(a)
		if err != nil {
			glog.Fatalf("couldn't generate auth file:", err)
		}
		if err = ioutil.WriteFile(fileName, auth, 0600); err != nil {
			glog.Fatalf("couldn't write auth file:", err)
		}
		return usr, passwd
	}
	j, err := ioutil.ReadFile(fileName)
	if err != nil {
		glog.Fatalf("couldn't read auth file:", err)
	}
	var a authFile
	if err = json.Unmarshal(j, &a); err != nil {
		glog.Fatalf("couldn't parse auth file:", err)
	}
	return a.User, a.Password
}

func rmHashComments(input []byte) []byte {
	rmCmnts, err := regexp.Compile(`(?m)^#.*\n`)
	if err != nil {
		glog.Warningf("regex failed to compile: %v", err)
		return input
	}
	return rmCmnts.ReplaceAllLiteral(input, []byte{})
}

func masterStartupScript(project, releaseName string) []byte {
	htpasswd := getHtpasswd()
	masterStartup := []byte(
		"#! /bin/bash\n" +
			"MASTER_NAME=" + MasterName + "\n" +
			"MASTER_RELEASE_TAR=" + releaseName + "\n" +
			"MASTER_HTPASSWD='" + htpasswd + "'\n")
	dr, err := ioutil.ReadFile("cluster/templates/download-release.sh")
	if err != nil {
		glog.Fatalf("failed to read download-release.sh: %v", err)
		return nil
	}
	sm, err := ioutil.ReadFile("cluster/templates/salt-master.sh")
	if err != nil {
		glog.Fatalf("failed to read salt-master.sh: %v", err)
		return nil
	}

	dr = rmHashComments(dr)
	sm = rmHashComments(sm)
	return bytes.Join([][]byte{masterStartup, dr, sm}, nil)
}

func minionStartupScript(i int) []byte {
	ipRange := fmt.Sprintf(minionIPs, i)
	minionStartup := []byte(
		"#! /bin/bash\n" +
			"MASTER_NAME=" + MasterName + "\n" +
			"MINION_IP_RANGE=" + ipRange + "\n")
	sm, err := ioutil.ReadFile("cluster/templates/salt-minion.sh")
	if err != nil {
		glog.Fatalf("failed to read salt-minion.sh: %v", err)
		return nil
	}
	sm = rmHashComments(sm)
	return bytes.Join([][]byte{minionStartup, sm}, nil)
}

func getHtpasswd() string {
	usr, passwd := GetCredentials()
	salt := randAlphaNumString(2, true)
	// This directly wraps the libc version of crypt
	data := C.struct_crypt_data{}
	cPasswd := C.CString(passwd)
	cSalt := C.CString(salt)
	htpasswd := C.GoString(C.crypt_r(cPasswd, cSalt, &data))
	C.free(unsafe.Pointer(cPasswd))
	C.free(unsafe.Pointer(cSalt))
	return fmt.Sprintf("%v:%v", usr, htpasswd)
}

func getReleaseName(gce *gce_cloud.GCECloud, project string) (string, error) {
	// TODO: retrieval of releases only works for the dev flow right now
	bucketHash := fmt.Sprintf("%x", md5.Sum([]byte(project)))[:5]
	u, err := user.Current()
	if err != nil {
		return "", err
	}
	tagName := fmt.Sprintf("gs://kubernetes-releases-%v/devel/%v/testing", bucketHash, u.Username)
	releaseNameBytes, err := readFromGCS(tagName)
	if err != nil {
		return "", err
	}
	release := strings.TrimSpace(string(releaseNameBytes)) + "/master-release.tgz"

	return release, nil
	// TODO: Check whether there is actually a release there (gce storage API)
}

// readFromGCS expects locations in the form gs://bucket/path/to/object
func readFromGCS(loc string) ([]byte, error) {
	c := gce_cloud.CreateOAuthClient()
	svc, err := storage.New(c)
	if err != nil {
		return nil, err
	}
	bucket, objName := parseGCSPath(loc)
	object, err := svc.Objects.Get(bucket, objName).Do()
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("GET", object.MediaLink, nil)
	if err != nil {
		return nil, err
	}
	req.URL.Path = strings.Replace(req.URL.Path, bucket, url.QueryEscape(bucket), 1)
	req.URL.Path = strings.Replace(req.URL.Path, objName, url.QueryEscape(objName), 1)
	googleapi.SetOpaque(req.URL)
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, errors.New(resp.Status)
	}

	contents, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return contents, nil
}

func parseGCSPath(loc string) (string, string) {
	if strings.EqualFold("gs://", loc[:5]) {
		loc = loc[5:]
	}
	i := strings.Index(loc, "/")
	if i == -1 {
		return "", ""
	}
	bucket := loc[:i]
	objName := loc[i+1:]
	return bucket, objName
}

type authFile struct {
	User     string
	Password string
}

func randAlphaNumString(n int, includePunc bool) string {
	alphanum := "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	if includePunc {
		alphanum = alphanum + "./"
	}
	var randString = make([]byte, n)
	for i := range randString {
		randString[i] = alphanum[rand.Int31n(int32(len(alphanum)))]
	}
	return string(randString)
}
