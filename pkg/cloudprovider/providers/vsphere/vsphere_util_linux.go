// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere

import (
	"fmt"
	"io/ioutil"
	"strings"
)

const (
	UUIDPath   = "/sys/class/dmi/id/product_serial"
	UUIDPrefix = "VMware-"
)

func GetVMUUID() (string, error) {
	id, err := ioutil.ReadFile(UUIDPath)
	if err != nil {
		return "", fmt.Errorf("error retrieving vm uuid: %s", err)
	}
	uuidFromFile := string(id[:])
	//strip leading and trailing white space and new line char
	uuid := strings.TrimSpace(uuidFromFile)
	// check the uuid starts with "VMware-"
	if !strings.HasPrefix(uuid, UUIDPrefix) {
		return "", fmt.Errorf("Failed to match Prefix, UUID read from the file is %v", uuidFromFile)
	}
	// Strip the prefix and white spaces and -
	uuid = strings.Replace(uuid[len(UUIDPrefix):(len(uuid))], " ", "", -1)
	uuid = strings.Replace(uuid, "-", "", -1)
	if len(uuid) != 32 {
		return "", fmt.Errorf("Length check failed, UUID read from the file is %v", uuidFromFile)
	}
	// need to add dashes, e.g. "564d395e-d807-e18a-cb25-b79f65eb2b9f"
	uuid = fmt.Sprintf("%s-%s-%s-%s-%s", uuid[0:8], uuid[8:12], uuid[12:16], uuid[16:20], uuid[20:32])
	return uuid, nil
}
