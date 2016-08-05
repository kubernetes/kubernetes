// Copyright 2016 Google Inc. All Rights Reserved.
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

package docker

import (
	"fmt"
	"os"
	"strings"

	dockertypes "github.com/docker/engine-api/types"
)

const (
	DockerInfoDriver         = "Driver"
	DockerInfoDriverStatus   = "DriverStatus"
	DriverStatusPoolName     = "Pool Name"
	DriverStatusDataLoopFile = "Data loop file"
	DriverStatusMetadataFile = "Metadata file"
)

func DriverStatusValue(status [][2]string, target string) string {
	for _, v := range status {
		if strings.EqualFold(v[0], target) {
			return v[1]
		}
	}

	return ""
}

func DockerThinPoolName(info dockertypes.Info) (string, error) {
	poolName := DriverStatusValue(info.DriverStatus, DriverStatusPoolName)
	if len(poolName) == 0 {
		return "", fmt.Errorf("Could not get devicemapper pool name")
	}

	return poolName, nil
}

func DockerMetadataDevice(info dockertypes.Info) (string, error) {
	metadataDevice := DriverStatusValue(info.DriverStatus, DriverStatusMetadataFile)
	if len(metadataDevice) != 0 {
		return metadataDevice, nil
	}

	poolName, err := DockerThinPoolName(info)
	if err != nil {
		return "", err
	}

	metadataDevice = fmt.Sprintf("/dev/mapper/%s_tmeta", poolName)

	if _, err := os.Stat(metadataDevice); err != nil {
		return "", err
	}

	return metadataDevice, nil
}
