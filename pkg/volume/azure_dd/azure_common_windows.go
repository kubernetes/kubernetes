// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package azure_dd

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util/mount"
)

func scsiHostRescan(io ioHandler, exec mount.Exec) {
	cmd := "Update-HostStorageCache"
	output, err := exec.Run("powershell", "/c", cmd)
	if err != nil {
		glog.Errorf("Update-HostStorageCache failed in scsiHostRescan, error: %v, output: %q", err, string(output))
	}
}

// search Windows disk number by LUN
func findDiskByLun(lun int, iohandler ioHandler, exec mount.Exec) (string, error) {
	cmd := `Get-Disk | select number, location | ConvertTo-Json`
	output, err := exec.Run("powershell", "/c", cmd)
	if err != nil {
		glog.Errorf("Get-Disk failed in findDiskByLun, error: %v, output: %q", err, string(output))
		return "", err
	}

	if len(output) < 10 {
		return "", fmt.Errorf("Get-Disk output is too short, output: %q", string(output))
	}

	var data []map[string]interface{}
	if err = json.Unmarshal(output, &data); err != nil {
		glog.Errorf("Get-Disk output is not a json array, output: %q", string(output))
		return "", err
	}

	for _, v := range data {
		if jsonLocation, ok := v["location"]; ok {
			if location, ok := jsonLocation.(string); ok {
				if !strings.Contains(location, " LUN ") {
					continue
				}

				arr := strings.Split(location, " ")
				arrLen := len(arr)
				if arrLen < 3 {
					glog.Warningf("unexpected json structure from Get-Disk, location: %q", jsonLocation)
					continue
				}

				glog.V(4).Infof("found a disk, locatin: %q, lun: %q", location, arr[arrLen-1])
				//last element of location field is LUN number, e.g.
				//		"location":  "Integrated : Adapter 3 : Port 0 : Target 0 : LUN 1"
				l, err := strconv.Atoi(arr[arrLen-1])
				if err != nil {
					glog.Warningf("cannot parse element from data structure, location: %q, element: %q", location, arr[arrLen-1])
					continue
				}

				if l == lun {
					glog.V(4).Infof("found a disk and lun, locatin: %q, lun: %d", location, lun)
					if d, ok := v["number"]; ok {
						if diskNum, ok := d.(float64); ok {
							glog.V(2).Infof("azureDisk Mount: got disk number(%d) by LUN(%d)", int(diskNum), lun)
							return strconv.Itoa(int(diskNum)), nil
						}
						glog.Warningf("LUN(%d) found, but could not get disk number(%q), location: %q", lun, d, location)
					}
					return "", fmt.Errorf("LUN(%d) found, but could not get disk number, location: %q", lun, location)
				}
			}
		}
	}

	return "", nil
}

func formatIfNotFormatted(disk string, fstype string, exec mount.Exec) {
	if err := mount.ValidateDiskNumber(disk); err != nil {
		glog.Errorf("azureDisk Mount: formatIfNotFormatted failed, err: %v\n", err)
		return
	}

	if len(fstype) == 0 {
		// Use 'NTFS' as the default
		fstype = "NTFS"
	}
	cmd := fmt.Sprintf("Get-Disk -Number %s | Where partitionstyle -eq 'raw' | Initialize-Disk -PartitionStyle MBR -PassThru", disk)
	cmd += fmt.Sprintf(" | New-Partition -AssignDriveLetter -UseMaximumSize | Format-Volume -FileSystem %s -Confirm:$false", fstype)
	output, err := exec.Run("powershell", "/c", cmd)
	if err != nil {
		glog.Errorf("azureDisk Mount: Get-Disk failed, error: %v, output: %q", err, string(output))
	} else {
		glog.Infof("azureDisk Mount: Disk successfully formatted, disk: %q, fstype: %q\n", disk, fstype)
	}
}
