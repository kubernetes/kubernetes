// Copyright 2019 The Prometheus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package iscsi

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// GetStats is the main iscsi status information func
// building the path and prepare info for enable iscsi
func GetStats(iqnPath string) (*Stats, error) {
	var istats Stats

	istats.Name = filepath.Base(iqnPath)
	istats.RootPath = filepath.Dir(iqnPath)

	matches, err := filepath.Glob(filepath.Join(iqnPath, "tpgt*"))
	if err != nil {
		return nil, fmt.Errorf("iscsi: GetStats: get TPGT path error %v", err)
	}
	istats.Tpgt = make([]TPGT, len(matches))

	for i, tpgtPath := range matches {
		istats.Tpgt[i].Name = filepath.Base(tpgtPath)
		istats.Tpgt[i].TpgtPath = tpgtPath
		istats.Tpgt[i].IsEnable, _ = isPathEnable(tpgtPath)
		if istats.Tpgt[i].IsEnable {
			matchesLunsPath, _ := filepath.Glob(filepath.Join(tpgtPath, "lun/lun*"))

			for _, lunPath := range matchesLunsPath {
				lun, err := getLunLinkTarget(lunPath)
				if err == nil {
					istats.Tpgt[i].Luns = append(istats.Tpgt[i].Luns, lun)
				}
			}
		}
	}
	return &istats, nil
}

// isPathEnable is a utility function
// check if the file "enable" contain enable message
func isPathEnable(path string) (bool, error) {
	enableReadout, err := ioutil.ReadFile(filepath.Join(path, "enable"))
	if err != nil {
		return false, fmt.Errorf("iscsi: isPathEnable ReadFile error %v", err)
	}
	isEnable, err := strconv.ParseBool(strings.TrimSpace(string(enableReadout)))
	if err != nil {
		return false, fmt.Errorf("iscsi: isPathEnable ParseBool error %v", err)
	}
	return isEnable, nil
}

func getLunLinkTarget(lunPath string) (lunObject LUN, err error) {
	lunObject.Name = filepath.Base(lunPath)
	lunObject.LunPath = lunPath
	files, err := ioutil.ReadDir(lunPath)
	if err != nil {
		return lunObject, fmt.Errorf("getLunLinkTarget: ReadDir path %s error %v", lunPath, err)
	}
	for _, file := range files {
		fileInfo, _ := os.Lstat(filepath.Join(lunPath, file.Name()))
		if fileInfo.Mode()&os.ModeSymlink != 0 {
			target, err := os.Readlink(filepath.Join(lunPath, fileInfo.Name()))
			if err != nil {
				return lunObject, fmt.Errorf("getLunLinkTarget: Readlink err %v", err)
			}
			targetPath, objectName := filepath.Split(target)
			_, typeWithNumber := filepath.Split(filepath.Clean(targetPath))

			underscore := strings.LastIndex(typeWithNumber, "_")

			if underscore != -1 {
				lunObject.Backstore = typeWithNumber[:underscore]
				lunObject.TypeNumber = typeWithNumber[underscore+1:]
			}

			lunObject.ObjectName = objectName
			return lunObject, nil
		}
	}
	return lunObject, errors.New("iscsi: getLunLinkTarget: Lun Link does not exist")
}

// ReadWriteOPS read and return the stat of read and write in megabytes,
// and total commands that send to the target
func ReadWriteOPS(iqnPath string, tpgt string, lun string) (readmb uint64,
	writemb uint64, iops uint64, err error) {

	readmbPath := filepath.Join(iqnPath, tpgt, "lun", lun,
		"statistics/scsi_tgt_port/read_mbytes")
	readmb, err = util.ReadUintFromFile(readmbPath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("iscsi: ReadWriteOPS: read_mbytes error file %s and %v",
			readmbPath, err)
	}

	writembPath := filepath.Join(iqnPath, tpgt, "lun", lun,
		"statistics/scsi_tgt_port/write_mbytes")
	writemb, err = util.ReadUintFromFile(writembPath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("iscsi: ReadWriteOPS: write_mbytes error file %s and %v",
			writembPath, err)
	}

	iopsPath := filepath.Join(iqnPath, tpgt, "lun", lun,
		"statistics/scsi_tgt_port/in_cmds")
	iops, err = util.ReadUintFromFile(iopsPath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("iscsi: ReadWriteOPS: in_cmds error file %s and %v",
			iopsPath, err)
	}

	return readmb, writemb, iops, nil
}

// GetFileioUdev is getting the actual info to build up
// the FILEIO data and match with the enable target
func (fs FS) GetFileioUdev(fileioNumber string, objectName string) (*FILEIO, error) {
	fileio := FILEIO{
		Name:       "fileio_" + fileioNumber,
		Fnumber:    fileioNumber,
		ObjectName: objectName,
	}
	udevPath := fs.configfs.Path(targetCore, fileio.Name, fileio.ObjectName, "udev_path")

	if _, err := os.Stat(udevPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("iscsi: GetFileioUdev: fileio_%s is missing file name", fileio.Fnumber)
	}
	filename, err := ioutil.ReadFile(udevPath)
	if err != nil {
		return nil, fmt.Errorf("iscsi: GetFileioUdev: Cannot read filename from udev link :%s", udevPath)
	}
	fileio.Filename = strings.TrimSpace(string(filename))

	return &fileio, nil
}

// GetIblockUdev is getting the actual info to build up
// the IBLOCK data and match with the enable target
func (fs FS) GetIblockUdev(iblockNumber string, objectName string) (*IBLOCK, error) {
	iblock := IBLOCK{
		Name:       "iblock_" + iblockNumber,
		Bnumber:    iblockNumber,
		ObjectName: objectName,
	}
	udevPath := fs.configfs.Path(targetCore, iblock.Name, iblock.ObjectName, "udev_path")

	if _, err := os.Stat(udevPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("iscsi: GetIBlockUdev: iblock_%s is missing file name", iblock.Bnumber)
	}
	filename, err := ioutil.ReadFile(udevPath)
	if err != nil {
		return nil, fmt.Errorf("iscsi: GetIBlockUdev: Cannot read iblock from udev link :%s", udevPath)
	}
	iblock.Iblock = strings.TrimSpace(string(filename))

	return &iblock, nil
}

// GetRBDMatch is getting the actual info to build up
// the RBD data and match with the enable target
func (fs FS) GetRBDMatch(rbdNumber string, poolImage string) (*RBD, error) {
	rbd := RBD{
		Name:    "rbd_" + rbdNumber,
		Rnumber: rbdNumber,
	}
	systemRbds, err := filepath.Glob(fs.sysfs.Path(devicePath, "[0-9]*"))
	if err != nil {
		return nil, fmt.Errorf("iscsi: GetRBDMatch: Cannot find any rbd block")
	}

	for systemRbdNumber, systemRbdPath := range systemRbds {
		var systemPool, systemImage string
		systemPoolPath := filepath.Join(systemRbdPath, "pool")
		if _, err := os.Stat(systemPoolPath); os.IsNotExist(err) {
			continue
		}
		bSystemPool, err := ioutil.ReadFile(systemPoolPath)
		if err != nil {
			continue
		} else {
			systemPool = strings.TrimSpace(string(bSystemPool))
		}

		systemImagePath := filepath.Join(systemRbdPath, "name")
		if _, err := os.Stat(systemImagePath); os.IsNotExist(err) {
			continue
		}
		bSystemImage, err := ioutil.ReadFile(systemImagePath)
		if err != nil {
			continue
		} else {
			systemImage = strings.TrimSpace(string(bSystemImage))
		}

		if strings.Compare(strconv.FormatInt(int64(systemRbdNumber), 10), rbdNumber) == 0 &&
			matchPoolImage(systemPool, systemImage, poolImage) {
			rbd.Pool = systemPool
			rbd.Image = systemImage
			return &rbd, nil
		}
	}
	return nil, nil
}

// GetRDMCPPath is getting the actual info to build up RDMCP data
func (fs FS) GetRDMCPPath(rdmcpNumber string, objectName string) (*RDMCP, error) {
	rdmcp := RDMCP{
		Name:       "rd_mcp_" + rdmcpNumber,
		ObjectName: objectName,
	}
	rdmcpPath := fs.configfs.Path(targetCore, rdmcp.Name, rdmcp.ObjectName)

	if _, err := os.Stat(rdmcpPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("iscsi: GetRDMCPPath: %s does not exist", rdmcpPath)
	}
	isEnable, err := isPathEnable(rdmcpPath)
	if err != nil {
		return nil, fmt.Errorf("iscsi: GetRDMCPPath: error %v", err)
	}
	if isEnable {
		return &rdmcp, nil
	}
	return nil, nil
}

func matchPoolImage(pool string, image string, matchPoolImage string) (isEqual bool) {
	var poolImage = fmt.Sprintf("%s-%s", pool, image)
	return strings.Compare(poolImage, matchPoolImage) == 0
}
