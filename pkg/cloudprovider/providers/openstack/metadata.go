/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
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

// metadataUrl is URL to OpenStack metadata server. It's hardcoded IPv4
// link-local address as documented in "OpenStack Cloud Administrator Guide",
// chapter Compute - Networking with nova-network.
// http://docs.openstack.org/admin-guide-cloud/compute-networking-nova.html#metadata-service
const metadataUrl = "http://169.254.169.254/openstack/2012-08-10/meta_data.json"

// Config drive is defined as an iso9660 or vfat (deprecated) drive
// with the "config-2" label.
// http://docs.openstack.org/user-guide/cli-config-drive.html
const configDriveLabel = "config-2"
const configDrivePath = "openstack/2012-08-10/meta_data.json"

var ErrBadMetadata = errors.New("Invalid OpenStack metadata, got empty uuid")

// Assumes the "2012-08-10" meta_data.json format.
// See http://docs.openstack.org/user-guide/cli_config_drive.html
type Metadata struct {
	Uuid             string `json:"uuid"`
	Name             string `json:"name"`
	AvailabilityZone string `json:"availability_zone"`
	// .. and other fields we don't care about.  Expand as necessary.
}

// parseMetadataUUID reads JSON from OpenStack metadata server and parses
// instance ID out of it.
func parseMetadata(r io.Reader) (*Metadata, error) {
	var metadata Metadata
	json := json.NewDecoder(r)
	if err := json.Decode(&metadata); err != nil {
		return nil, err
	}

	if metadata.Uuid == "" {
		return nil, ErrBadMetadata
	}

	return &metadata, nil
}

func getMetadataFromConfigDrive() (*Metadata, error) {
	// Try to read instance UUID from config drive.
	dev := "/dev/disk/by-label/" + configDriveLabel
	if _, err := os.Stat(dev); os.IsNotExist(err) {
		out, err := exec.New().Command(
			"blkid", "-l",
			"-t", "LABEL="+configDriveLabel,
			"-o", "device",
		).CombinedOutput()
		if err != nil {
			glog.V(2).Infof("Unable to run blkid: %v", err)
			return nil, err
		}
		dev = strings.TrimSpace(string(out))
	}

	mntdir, err := ioutil.TempDir("", "configdrive")
	if err != nil {
		return nil, err
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
		return nil, err
	}
	defer mounter.Unmount(mntdir)

	glog.V(4).Infof("Configdrive mounted on %s", mntdir)

	f, err := os.Open(
		filepath.Join(mntdir, configDrivePath))
	if err != nil {
		glog.Errorf("Error reading %s on config drive: %v", configDrivePath, err)
		return nil, err
	}
	defer f.Close()

	return parseMetadata(f)
}

func getMetadataFromMetadataService() (*Metadata, error) {
	// Try to get JSON from metdata server.
	glog.V(4).Infof("Attempting to fetch metadata from %s", metadataUrl)
	resp, err := http.Get(metadataUrl)
	if err != nil {
		glog.V(3).Infof("Cannot read %s: %v", metadataUrl, err)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		err = fmt.Errorf("Unexpected status code when reading metadata from %s: %s", metadataUrl, resp.Status)
		glog.V(3).Infof("%v", err)
		return nil, err
	}

	return parseMetadata(resp.Body)
}

// Metadata is fixed for the current host, so cache the value process-wide
var metadataCache *Metadata

func getMetadata() (*Metadata, error) {
	if metadataCache == nil {
		md, err := getMetadataFromConfigDrive()
		if err != nil {
			md, err = getMetadataFromMetadataService()
		}
		if err != nil {
			return nil, err
		}
		metadataCache = md
	}
	return metadataCache, nil
}
