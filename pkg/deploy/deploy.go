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
package deploy

import (
	"bytes"
	"crypto/md5"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/exec"
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
	masterName     = instancePrefix + "-master"
	minionPrefix   = instancePrefix + "-minion"
	masterTag      = masterName
	minionTag      = minionPrefix
	masterSize     = "g1-small"
	minionSize     = "g1-small"
	image          = "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/backports-debian-7-wheezy-v20140814"
	minionIPs      = "10.244.%v.0/24"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// DeployMaster creates the firewall rules and instance for the kube master.
func DeployMaster(gce *gce_cloud.GCECloud) ([]*gce_cloud.GCEOp, error) {
	var ops []*gce_cloud.GCEOp
	glog.Info("Creating firewall kubernetes-master-https\n")
	op, err := createMasterFirewall(gce)
	if err != nil {
		return ops, fmt.Errorf("couldn't start operation: %v", err)
	}
	ops = append(ops, op)
	glog.Info("Creating instance kubernetes-master\n")
	op, err = createMaster(gce)
	if err != nil {
		return ops, fmt.Errorf("couldn't start operation: %v", err)
	}
	ops = append(ops, op)
	return ops, nil
}

// DeployMinion creates the firewall rules, instance, and route for the specified minion.
func DeployMinion(cloud *gce_cloud.GCECloud, i int) ([]*gce_cloud.GCEOp, error) {
	var ops []*gce_cloud.GCEOp
	glog.Infof("Creating firewall kubernetes-minion-%v-all\n", i)
	op, err := createMinionFirewall(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't create firewall-rule insert operation: %v", err)
	}
	ops = append(ops, op)
	glog.Infof("Creating instance kubernetes-minion-%v\n", i)
	op, err = createMinion(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't create instance insert operation: %v", err)
	}
	ops = append(ops, op)
	glog.Infof("Creating route kubernetes-minion-%v\n", i)
	op, err = createMinionRoute(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't create route insert operation: %v", err)
	}
	ops = append(ops, op)
	return ops, nil
}

// DownMaster removes the firewall rules and instance for the kube master.
func DownMaster(gce *gce_cloud.GCECloud) ([]*gce_cloud.GCEOp, error) {
	var ops []*gce_cloud.GCEOp
	glog.Info("Removing firewall kubernetes-master-https\n")
	op, err := deleteMasterFirewall(gce)
	if err != nil {
		return ops, fmt.Errorf("couldn't start operation: %v", err)
	}
	ops = append(ops, op)
	glog.Info("Removing instance kubernetes-master\n")
	op, err = deleteMaster(gce)
	if err != nil {
		return ops, fmt.Errorf("couldn't start operation: %v", err)
	}
	ops = append(ops, op)
	return ops, nil
}

// DeployMinion removes the firewall rules, instance, and route for the specified minion.
func DownMinion(cloud *gce_cloud.GCECloud, i int) ([]*gce_cloud.GCEOp, error) {
	var ops []*gce_cloud.GCEOp
	glog.Infof("Removing firewall kubernetes-minion-%v-all\n", i)
	op, err := deleteMinionFirewall(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't remove firewall-rule insert operation: %v", err)
	}
	ops = append(ops, op)
	glog.Infof("Removing instance kubernetes-minion-%v\n", i)
	op, err = deleteMinion(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't remove instance insert operation: %v", err)
	}
	ops = append(ops, op)
	glog.Infof("Removing route kubernetes-minion-%v\n", i)
	op, err = deleteMinionRoute(cloud, i)
	if err != nil {
		return ops, fmt.Errorf("couldn't remove route insert operation: %v", err)
	}
	ops = append(ops, op)
	return ops, nil
}

// WaitForOps polls the status of the specified operations until they are all "DONE"
func WaitForOps(cloud *gce_cloud.GCECloud, ops []*gce_cloud.GCEOp) error {
	// Wait for all operations to complete
	for _, op := range ops {
		op, err := cloud.PollOp(op)
		if err != nil {
			return err
		}
		for op.Status() != "DONE" {
			glog.Infof("Waiting 2s for %v of %v %v\n", op.OperationType(), op.Resource(), op.Target())
			time.Sleep(2 * time.Second)
			op, err = cloud.PollOp(op)
			if err != nil {
				return err
			}
		}
		if op.Errors() != nil {
			return errors.New("errors in operation:\n" + strings.Join(op.Errors(), "\n"))
		}
		glog.Infof("%v of %v %v has completed\n", op.OperationType(), op.Resource(), op.Target())
	}
	return nil
}

// CheckMaster attempts to contact the api-server on the master until it successfully responds
func CheckMaster(cloud *gce_cloud.GCECloud) error {
	kubeMasterIP, err := cloud.IPAddress(masterName)
	if err != nil {
		return fmt.Errorf("error getting master IP: %v\n", err)
	}
	glog.Infof("Using master: %v (external IP: %v)\n", masterName, kubeMasterIP)
	url := fmt.Sprintf("https://%v/api/v1beta1/pods", kubeMasterIP)
	usr, pass := getCredentials()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.SetBasicAuth(usr, pass)
	tr := &http.Transport{
		ResponseHeaderTimeout: 5 * time.Second,
		TLSClientConfig:       &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	glog.Info("Waiting for cluster initialization.\n")
	glog.Info("  This will continually check to see if the API for kubernetes is reachable.\n")
	glog.Info("  This might loop forever if there was some uncaught error during start up.\n")
	t := 0
	for {
		resp, err := client.Do(req)
		if err == nil {
			if resp.StatusCode != 200 {
				glog.Infof("\nResponse status was %v. Something might be wrong.\n", resp.Status)
			}
			break
		}
		time.Sleep(2 * time.Second)
		t = t + 2
		if t%10 == 0 {
			glog.Infof("%v seconds elapsed", t)
		}
	}
	glog.Info("Kubernetes master is running.  Access at:\n")
	glog.Infof("  https://%v:%v@%v\n", usr, pass, kubeMasterIP)
	return nil
}

// CheckMinions ssh'es into each minion and verifies that docker installed successfully.
func CheckMinions(n int) error {
	glog.Info("Sanity checking minions...\n")
	for i := 1; i <= n; i++ {
		name := fmt.Sprintf("%v-%v", minionPrefix, i)
		if err := exec.Command("gcutil", "ssh", name, "which", "docker").Run(); err != nil {
			return fmt.Errorf("Docker failed to install on %v. You're cluster is unlikely to work correctly.\n"+
				"Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)\n", name)

		}
	}
	return nil
}

// createMasterFirewall creates a firewall rule to allow https to the kube-master from anywhere.
func createMasterFirewall(gce *gce_cloud.GCECloud) (*gce_cloud.GCEOp, error) {
	return gce.CreateFirewall(masterName+"-https", "0.0.0.0/0", masterTag, "tcp:443")
}

// createMinionFirewall creates a firewall rule to allow communication on the internal IPs.
func createMinionFirewall(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v-all", minionPrefix, i)
	ipRange := fmt.Sprintf(minionIPs, i)
	return gce.CreateFirewall(name, ipRange, minionTag, "tcp,udp,icmp,esp,ah,sctp")
}

// createMaster spins up the kube-master instance.
func createMaster(gce *gce_cloud.GCECloud) (*gce_cloud.GCEOp, error) {
	release, err := getReleaseName(gce)
	if err != nil {
		return nil, err
	}
	masterStartup := masterStartupScript(gce.ProjectID(), release)
	scopes := []string{
		compute.DevstorageRead_onlyScope,
		compute.ComputeScope,
	}
	return gce.CreateInstance(masterName, masterSize, image, masterTag, string(masterStartup), false, scopes)
}

// createMinion spins up the specified minion instance.
func createMinion(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v", minionPrefix, i)
	minionStartup := minionStartupScript(i)
	return gce.CreateInstance(name, minionSize, image, minionTag, string(minionStartup), true, []string{})
}

// createMinionRoute adds the minion routing rule to allow forwarding to the pods.
func createMinionRoute(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v", minionPrefix, i)
	nextHop := fmt.Sprintf("%v/zones/%v/instances/%v", gce_cloud.FullQualProj(gce.ProjectID()), gce.Zone(), name)
	ipRange := fmt.Sprintf("10.244.%v.0/24", i)
	return gce.CreateRoute(name, nextHop, ipRange)
}

// deleteMasterFirewall removes the master firewall rule.
func deleteMasterFirewall(gce *gce_cloud.GCECloud) (*gce_cloud.GCEOp, error) {
	return gce.DeleteFirewall(masterName + "-https")
}

// deleteMinionFirewall removes the specified minion firewall rule.
func deleteMinionFirewall(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v-all", minionPrefix, i)
	return gce.DeleteFirewall(name)
}

// deleteMaster removes the kube-master instance.
func deleteMaster(gce *gce_cloud.GCECloud) (*gce_cloud.GCEOp, error) {
	return gce.DeleteInstance(masterName)
}

// deleteMinion removes the specified minion instance.
func deleteMinion(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v", minionPrefix, i)
	return gce.DeleteInstance(name)
}

// deleteMinionRoute removes the minion routing rule.
func deleteMinionRoute(gce *gce_cloud.GCECloud, i int) (*gce_cloud.GCEOp, error) {
	name := fmt.Sprintf("%v-%v", minionPrefix, i)
	return gce.DeleteRoute(name)
}

// getCredentials gets the credentials stored in the .kubernetes_auth file (or create and store new credentials).
func getCredentials() (string, string) {
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
			"MASTER_NAME=" + masterName + "\n" +
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
			"MASTER_NAME=" + masterName + "\n" +
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
	usr, passwd := getCredentials()
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

func getReleaseName(gce *gce_cloud.GCECloud) (string, error) {
	// TODO: retrieval of releases only works for the dev flow right now
	bucketHash := fmt.Sprintf("%x", md5.Sum([]byte(gce.ProjectID())))[:5]
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
