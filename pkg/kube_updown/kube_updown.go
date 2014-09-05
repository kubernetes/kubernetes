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
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"

	compute "code.google.com/p/google-api-go-client/compute/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	"github.com/golang/glog"
)

const (
	instancePrefix = "kubernetes"
	MasterName     = instancePrefix + "-master"
	MinionPrefix   = instancePrefix + "-minion"
	masterTag      = MasterName
	minionTag      = MinionPrefix
	masterSize     = "g1-small"
	minionSize     = "g1-small"
	image          = "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/backports-debian-7-wheezy-v20140814"
)

// CreateMasterFirewall creates a firewall rule to allow https to the kube-master from anywhere.
func CreateMasterFirewall(gce *gce_cloud.GCECloud) (*gce_cloud.GCEOp, error) {
	return gce.CreateFirewall(MasterName+"-https", "0.0.0.0/0", masterTag, "tcp:443")
}

// CreateMinionFirewall creates a firewall rule to allow communication on the internal IPs.
func CreateMinionFirewall(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v-all", MinionPrefix, i)
	ipRange := fmt.Sprintf("10.244.%v.0/24", i)
	return gce.CreateFirewall(name, ipRange, minionTag, "tcp,udp,icmp,esp,ah,sctp")
}

// CreateMaster spins up the kube-master instance.
func CreateMaster(gce *gce_cloud.GCECloud, project string) (*gce_cloud.GCEOp, error) {
	masterStartup := masterStartupScript(project)
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
	nextHop := fmt.Sprintf("%v/zones/%v/instances/%v-%v", fqProj(project), zone, MinionPrefix, i)
	ipRange := fmt.Sprintf("10.244.%v.0/24", i)
	return gce.CreateRoute(name, nextHop, ipRange)
}

// GetMasterIP gets the IP address of the cluster's master.
func GetMasterIP(svc *compute.Service, project, zone string) (string, error) {
	inst, err := svc.Instances.Get(project, zone, MasterName).Do()
	if err != nil {
		return "", err
	}
	nwIntf := inst.NetworkInterfaces[0]
	if nwIntf == nil {
		return "", errors.New("No network interfaces found")
	}
	axCfg := nwIntf.AccessConfigs[0]
	if axCfg == nil {
		return "", errors.New("No access configurations found")
	}
	ip := axCfg.NatIP
	return ip, nil
}

// GetCredentials gets the credentials stored in the .kubernetes_auth file (or create and store new credentials).
func GetCredentials() (string, string) {
	usr, err := user.Current()
	var homeDir string
	if err != nil {
		homeDir = "."
	}
	homeDir = usr.HomeDir
	fileName := filepath.Join(homeDir, ".kubernetes_auth")
	if _, err := os.Stat(fileName); os.IsNotExist(err) {
		usr := "admin"
		passwd := randAlphaNumString(16)
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

func addFirewall(svc *compute.Service, project, name, sourceRange, tag string, allowed []*compute.FirewallAllowed) (*compute.Operation, error) {
	prefix := fqProj(project)
	firewall := &compute.Firewall{
		Name:    name,
		Network: prefix + "/global/networks/default",
		Allowed: allowed,
		SourceRanges: []string{
			sourceRange,
		},
		TargetTags: []string{
			tag,
		},
	}

	return svc.Firewalls.Insert(project, firewall).Do()
}

func rmHashComments(input []byte) []byte {
	rmCmnts, err := regexp.Compile(`(?m)^#.*\n`)
	if err != nil {
		glog.Warningf("regex failed to compile: %v", err)
		return input
	}
	return rmCmnts.ReplaceAllLiteral(input, []byte{})
}

func masterStartupScript(project string) []byte {
	htpasswd := getHtpasswd()
	releaseName := getReleaseName(project)
	masterStartup := []byte(
		"#! /bin/bash\n" +
			"MASTER_NAME=" + MasterName + "\n" +
			"MASTER_RELEASE_TAR=" + releaseName + "\n" +
			"MASTER_HTPASSWD='" + strings.TrimSpace(string(htpasswd)) + "'\n")
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
	ipRange := fmt.Sprintf("10.244.%v.0/24", i)
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

func getHtpasswd() []byte {
	kubeTemp, err := ioutil.TempDir(os.TempDir(), "kubernetes")
	if err != nil {
		glog.Fatalf("failed to create temp directory: %v", err)
		return nil
	}
	defer os.RemoveAll(kubeTemp)
	usr, passwd := GetCredentials()
	glog.Infof("Using password: %v:%v\n", usr, passwd)
	// TODO: kube-master auth (this will break if run out of diff dir)
	hPy, _ := filepath.Abs("third_party/htpasswd/htpasswd.py")
	hFile := filepath.Join(kubeTemp, "htpasswd")
	err = exec.Command("python", hPy, "-b", "-c", hFile, usr, passwd).Run()
	if err != nil {
		glog.Fatalf("failed to execute htpasswd: %v", err)
		return nil
	}
	htpasswd, err := ioutil.ReadFile(hFile)
	if err != nil {
		glog.Fatalf("failed to read htpasswd file: %v", err)
		return nil
	}
	return htpasswd
}

func getReleaseName(project string) string {
	// TODO: retrieval of releases only works for the dev flow right now
	bucketHash := fmt.Sprintf("%x", md5.Sum([]byte(project)))[:5]
	u, err := user.Current()
	if err != nil {
		glog.Fatalf("failed to obtain current user: %v", err)
		return ""
	}
	tagName := fmt.Sprintf("gs://kubernetes-releases-%v/devel/%v/testing", bucketHash, u.Username)
	releaseNameBytes, err := readFromGsUrl(tagName)
	if err != nil {
		glog.Fatalf("failed to read from Google Storage url: %v", err)
		return ""
	}
	return strings.TrimSpace(string(releaseNameBytes)) + "/master-release.tgz"
	// TODO: Check whether there is actually a release there (gce storage API)
}

func readFromGsUrl(url string) ([]byte, error) {
	cmd := exec.Command("gsutil", "-q", "cat", url)
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		return nil, err
	}
	contents, err := ioutil.ReadAll(outPipe)
	if err != nil {
		return nil, err
	}
	if err = cmd.Wait(); err != nil {
		return nil, err
	}
	return contents, nil
}

func toMetadataItems(m map[string]string) []*compute.MetadataItems {
	mdi := make([]*compute.MetadataItems, len(m))
	for k, v := range m {
		mdi[0] = &compute.MetadataItems{
			Key:   k,
			Value: v,
		}
	}
	return mdi
}

func fqProj(proj string) string {
	return "https://www.googleapis.com/compute/v1/projects/" + proj
}

type authFile struct {
	User     string
	Password string
}

func randAlphaNumString(n int) string {
	const alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	var randString = make([]byte, n)
	for i := range randString {
		randString[i] = alphanum[rand.Int31n(int32(len(alphanum)))]
	}
	return string(randString)
}
