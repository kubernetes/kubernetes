/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package openstack

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
)

// Assumes the "2012-08-10" meta_data.json format.
// See http://docs.openstack.org/user-guide/cli_config_drive.html
type Metadata struct {
	Uuid string `json:"uuid"`
	Name string `json:"name"`
	// .. and other fields we don't care about.  Expand as necessary.
}

func parseMetadata(r io.Reader) (Metadata, error) {
	var metadata Metadata

	json := json.NewDecoder(r)
	err := json.Decode(&metadata)

	return metadata, err
}

func getMetadataFromConfigDrive() (Metadata, error) {
	// Try to read instance UUID from config drive.
	dev := "/dev/disk/by-label/config-2"
	if _, err := os.Stat(dev); os.IsNotExist(err) {
		out, err := exec.New().Command(
			"blkid", "-l",
			"-t", "LABEL=config-2",
			"-o", "device",
		).CombinedOutput()
		if err != nil {
			glog.V(2).Infof("Unable to run blkid: %v", err)
			return Metadata{}, err
		}
		dev = strings.TrimSpace(string(out))
	}

	mntdir, err := ioutil.TempDir("", "configdrive")
	if err != nil {
		return Metadata{}, err
	}
	defer os.Remove(mntdir)

	glog.V(4).Infof("Attempting to mount configdrive %s on %s", dev, mntdir)

	mounter := mount.New()
	err = mounter.Mount(dev, mntdir, "iso9660", []string{"ro"})
	if err != nil {
		err = mounter.Mount(dev, mntdir, "vfat", []string{"ro"})
	}
	if err != nil {
		glog.Errorf("Error mounting configdrive %s: %v", dev, err)
		return Metadata{}, err
	}
	defer mounter.Unmount(mntdir)

	glog.V(4).Infof("Configdrive mounted on %s", mntdir)

	f, err := os.Open(
		filepath.Join(mntdir, "openstack/2012-08-10/meta_data.json"))
	if err != nil {
		glog.Errorf("Error reading openstack/2012-08-10/meta_data.json on config drive: %v", err)
		return Metadata{}, err
	}
	defer f.Close()

	return parseMetadata(f)
}

func getMetadataFromMetadataService() (Metadata, error) {
	// Try to read instance UUID from metadata service
	const url = "http://169.254.169.254/openstack/2012-08-10/meta_data.json"

	glog.V(2).Infof("Fetching metadata from %s", url)

	resp, err := http.Get(url)
	if err != nil {
		glog.Errorf("Failed to get metadata from %s: %v", url, err)
		return Metadata{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		glog.Errorf("Metadata service returned %s", resp.Status)
		return Metadata{}, ErrHttpError
	}

	return parseMetadata(resp.Body)
}

func getMetadata() (Metadata, error) {
	md, err := getMetadataFromConfigDrive()
	if err != nil {
		md, err = getMetadataFromMetadataService()
	}
	return md, err
}
