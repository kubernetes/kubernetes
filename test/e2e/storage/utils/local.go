/*
Copyright 2019 The Kubernetes Authors.

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

package utils

/*
 * Various local test resource implementations.
 */

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
)

// LocalVolumeType represents type of local volume, e.g. tmpfs, directory,
// block, etc.
type LocalVolumeType string

const (
	// LocalVolumeDirectory reprensents a simple directory as local volume
	LocalVolumeDirectory LocalVolumeType = "dir"
	// LocalVolumeDirectoryLink is like LocalVolumeDirectory but it's a symbolic link to directory
	LocalVolumeDirectoryLink LocalVolumeType = "dir-link"
	// LocalVolumeDirectoryBindMounted is like LocalVolumeDirectory but bind mounted
	LocalVolumeDirectoryBindMounted LocalVolumeType = "dir-bindmounted"
	// LocalVolumeDirectoryLinkBindMounted is like LocalVolumeDirectory but it's a symbolic link to self bind mounted directory
	// Note that bind mounting at symbolic link actually mounts at directory it
	// links to
	LocalVolumeDirectoryLinkBindMounted LocalVolumeType = "dir-link-bindmounted"
	// LocalVolumeTmpfs represents a temporary filesystem to be used as local volume
	LocalVolumeTmpfs LocalVolumeType = "tmpfs"
	// LocalVolumeBlock represents a Block device, creates a local file, and maps it as a block device
	LocalVolumeBlock LocalVolumeType = "block"
	// LocalVolumeBlockFS represents a filesystem backed by a block device
	LocalVolumeBlockFS LocalVolumeType = "blockfs"
	// LocalVolumeGCELocalSSD represents a Filesystem backed by GCE Local SSD as local volume
	LocalVolumeGCELocalSSD LocalVolumeType = "gce-localssd-scsi-fs"
)

// LocalTestResource represents test resource of a local volume.
type LocalTestResource struct {
	VolumeType LocalVolumeType
	Node       *v1.Node
	// Volume path, path to filesystem or block device on the node
	Path string
	// If volume is backed by a loop device, we create loop device storage file
	// under this directory.
	loopDir string
}

// LocalTestResourceManager represents interface to create/destroy local test resources on node
type LocalTestResourceManager interface {
	Create(node *v1.Node, volumeType LocalVolumeType, parameters map[string]string) *LocalTestResource
	ExpandBlockDevice(ltr *LocalTestResource, mbToAdd int) error
	Remove(ltr *LocalTestResource)
}

// ltrMgr implements LocalTestResourceManager
type ltrMgr struct {
	prefix   string
	hostExec HostExec
	// hostBase represents a writable directory on the host under which we
	// create test directories
	hostBase string
}

// NewLocalResourceManager returns a instance of LocalTestResourceManager
func NewLocalResourceManager(prefix string, hostExec HostExec, hostBase string) LocalTestResourceManager {
	return &ltrMgr{
		prefix:   prefix,
		hostExec: hostExec,
		hostBase: hostBase,
	}
}

// getTestDir returns a test dir under `hostBase` directory with randome name.
func (l *ltrMgr) getTestDir() string {
	testDirName := fmt.Sprintf("%s-%s", l.prefix, string(uuid.NewUUID()))
	return filepath.Join(l.hostBase, testDirName)
}

func (l *ltrMgr) setupLocalVolumeTmpfs(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	ginkgo.By(fmt.Sprintf("Creating tmpfs mount point on node %q at path %q", node.Name, hostDir))
	err := l.hostExec.IssueCommand(fmt.Sprintf("mkdir -p %q && mount -t tmpfs -o size=10m tmpfs-%q %q", hostDir, hostDir, hostDir), node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeTmpfs(ltr *LocalTestResource) {
	ginkgo.By(fmt.Sprintf("Unmount tmpfs mount point on node %q at path %q", ltr.Node.Name, ltr.Path))
	err := l.hostExec.IssueCommand(fmt.Sprintf("umount %q", ltr.Path), ltr.Node)
	framework.ExpectNoError(err)

	ginkgo.By("Removing the test directory")
	err = l.hostExec.IssueCommand(fmt.Sprintf("rm -r %s", ltr.Path), ltr.Node)
	framework.ExpectNoError(err)
}

// createAndSetupLoopDevice creates an empty file and associates a loop devie with it.
func (l *ltrMgr) createAndSetupLoopDevice(dir string, node *v1.Node, size int) {
	ginkgo.By(fmt.Sprintf("Creating block device on node %q using path %q", node.Name, dir))
	mkdirCmd := fmt.Sprintf("mkdir -p %s", dir)
	count := size / 4096
	// xfs requires at least 4096 blocks
	if count < 4096 {
		count = 4096
	}
	ddCmd := fmt.Sprintf("dd if=/dev/zero of=%s/file bs=4096 count=%d", dir, count)
	losetupCmd := fmt.Sprintf("losetup -f %s/file", dir)
	err := l.hostExec.IssueCommand(fmt.Sprintf("%s && %s && %s", mkdirCmd, ddCmd, losetupCmd), node)
	framework.ExpectNoError(err)
}

// findLoopDevice finds loop device path by its associated storage directory.
func (l *ltrMgr) findLoopDevice(dir string, node *v1.Node) string {
	cmd := fmt.Sprintf("E2E_LOOP_DEV=$(losetup | grep %s/file | awk '{ print $1 }') 2>&1 > /dev/null && echo ${E2E_LOOP_DEV}", dir)
	loopDevResult, err := l.hostExec.IssueCommandWithResult(cmd, node)
	framework.ExpectNoError(err)
	return strings.TrimSpace(loopDevResult)
}

func (l *ltrMgr) setupLocalVolumeBlock(node *v1.Node, parameters map[string]string) *LocalTestResource {
	loopDir := l.getTestDir()
	l.createAndSetupLoopDevice(loopDir, node, 20*1024*1024)
	loopDev := l.findLoopDevice(loopDir, node)
	return &LocalTestResource{
		Node:    node,
		Path:    loopDev,
		loopDir: loopDir,
	}
}

// teardownLoopDevice tears down loop device by its associated storage directory.
func (l *ltrMgr) teardownLoopDevice(dir string, node *v1.Node) {
	loopDev := l.findLoopDevice(dir, node)
	ginkgo.By(fmt.Sprintf("Tear down block device %q on node %q at path %s/file", loopDev, node.Name, dir))
	losetupDeleteCmd := fmt.Sprintf("losetup -d %s", loopDev)
	err := l.hostExec.IssueCommand(losetupDeleteCmd, node)
	framework.ExpectNoError(err)
	return
}

func (l *ltrMgr) cleanupLocalVolumeBlock(ltr *LocalTestResource) {
	l.teardownLoopDevice(ltr.loopDir, ltr.Node)
	ginkgo.By(fmt.Sprintf("Removing the test directory %s", ltr.loopDir))
	removeCmd := fmt.Sprintf("rm -r %s", ltr.loopDir)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeBlockFS(node *v1.Node, parameters map[string]string) *LocalTestResource {
	ltr := l.setupLocalVolumeBlock(node, parameters)
	loopDev := ltr.Path
	loopDir := ltr.loopDir
	// Format and mount at loopDir and give others rwx for read/write testing
	cmd := fmt.Sprintf("mkfs -t ext4 %s && mount -t ext4 %s %s && chmod o+rwx %s", loopDev, loopDev, loopDir, loopDir)
	err := l.hostExec.IssueCommand(cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node:    node,
		Path:    loopDir,
		loopDir: loopDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeBlockFS(ltr *LocalTestResource) {
	umountCmd := fmt.Sprintf("umount %s", ltr.Path)
	err := l.hostExec.IssueCommand(umountCmd, ltr.Node)
	framework.ExpectNoError(err)
	l.cleanupLocalVolumeBlock(ltr)
}

func (l *ltrMgr) setupLocalVolumeDirectory(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	mkdirCmd := fmt.Sprintf("mkdir -p %s", hostDir)
	err := l.hostExec.IssueCommand(mkdirCmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectory(ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", ltr.Path)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeDirectoryLink(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	hostDirBackend := hostDir + "-backend"
	cmd := fmt.Sprintf("mkdir %s && ln -s %s %s", hostDirBackend, hostDirBackend, hostDir)
	err := l.hostExec.IssueCommand(cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryLink(ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	hostDir := ltr.Path
	hostDirBackend := hostDir + "-backend"
	removeCmd := fmt.Sprintf("rm -r %s && rm -r %s", hostDir, hostDirBackend)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeDirectoryBindMounted(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	cmd := fmt.Sprintf("mkdir %s && mount --bind %s %s", hostDir, hostDir, hostDir)
	err := l.hostExec.IssueCommand(cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryBindMounted(ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	hostDir := ltr.Path
	removeCmd := fmt.Sprintf("umount %s && rm -r %s", hostDir, hostDir)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeDirectoryLinkBindMounted(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	hostDirBackend := hostDir + "-backend"
	cmd := fmt.Sprintf("mkdir %s && mount --bind %s %s && ln -s %s %s", hostDirBackend, hostDirBackend, hostDirBackend, hostDirBackend, hostDir)
	err := l.hostExec.IssueCommand(cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryLinkBindMounted(ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	hostDir := ltr.Path
	hostDirBackend := hostDir + "-backend"
	removeCmd := fmt.Sprintf("rm %s && umount %s && rm -r %s", hostDir, hostDirBackend, hostDirBackend)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeGCELocalSSD(node *v1.Node, parameters map[string]string) *LocalTestResource {
	res, err := l.hostExec.IssueCommandWithResult("ls /mnt/disks/by-uuid/google-local-ssds-scsi-fs/", node)
	framework.ExpectNoError(err)
	dirName := strings.Fields(res)[0]
	hostDir := "/mnt/disks/by-uuid/google-local-ssds-scsi-fs/" + dirName
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeGCELocalSSD(ltr *LocalTestResource) {
	// This filesystem is attached in cluster initialization, we clean all files to make it reusable.
	removeCmd := fmt.Sprintf("find '%s' -mindepth 1 -maxdepth 1 -print0 | xargs -r -0 rm -rf", ltr.Path)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) expandLocalVolumeBlockFS(ltr *LocalTestResource, mbToAdd int) error {
	ddCmd := fmt.Sprintf("dd if=/dev/zero of=%s/file conv=notrunc oflag=append bs=1M count=%d", ltr.loopDir, mbToAdd)
	loopDev := l.findLoopDevice(ltr.loopDir, ltr.Node)
	losetupCmd := fmt.Sprintf("losetup -c %s", loopDev)
	return l.hostExec.IssueCommand(fmt.Sprintf("%s && %s", ddCmd, losetupCmd), ltr.Node)
}

func (l *ltrMgr) ExpandBlockDevice(ltr *LocalTestResource, mbtoAdd int) error {
	switch ltr.VolumeType {
	case LocalVolumeBlockFS:
		return l.expandLocalVolumeBlockFS(ltr, mbtoAdd)
	}
	return fmt.Errorf("Failed to expand local test resource, unsupported volume type: %s", ltr.VolumeType)
}

func (l *ltrMgr) Create(node *v1.Node, volumeType LocalVolumeType, parameters map[string]string) *LocalTestResource {
	var ltr *LocalTestResource
	switch volumeType {
	case LocalVolumeDirectory:
		ltr = l.setupLocalVolumeDirectory(node, parameters)
	case LocalVolumeDirectoryLink:
		ltr = l.setupLocalVolumeDirectoryLink(node, parameters)
	case LocalVolumeDirectoryBindMounted:
		ltr = l.setupLocalVolumeDirectoryBindMounted(node, parameters)
	case LocalVolumeDirectoryLinkBindMounted:
		ltr = l.setupLocalVolumeDirectoryLinkBindMounted(node, parameters)
	case LocalVolumeTmpfs:
		ltr = l.setupLocalVolumeTmpfs(node, parameters)
	case LocalVolumeBlock:
		ltr = l.setupLocalVolumeBlock(node, parameters)
	case LocalVolumeBlockFS:
		ltr = l.setupLocalVolumeBlockFS(node, parameters)
	case LocalVolumeGCELocalSSD:
		ltr = l.setupLocalVolumeGCELocalSSD(node, parameters)
	default:
		framework.Failf("Failed to create local test resource on node %q, unsupported volume type: %v is specified", node.Name, volumeType)
		return nil
	}
	if ltr == nil {
		framework.Failf("Failed to create local test resource on node %q, volume type: %v, parameters: %v", node.Name, volumeType, parameters)
	}
	ltr.VolumeType = volumeType
	return ltr
}

func (l *ltrMgr) Remove(ltr *LocalTestResource) {
	switch ltr.VolumeType {
	case LocalVolumeDirectory:
		l.cleanupLocalVolumeDirectory(ltr)
	case LocalVolumeDirectoryLink:
		l.cleanupLocalVolumeDirectoryLink(ltr)
	case LocalVolumeDirectoryBindMounted:
		l.cleanupLocalVolumeDirectoryBindMounted(ltr)
	case LocalVolumeDirectoryLinkBindMounted:
		l.cleanupLocalVolumeDirectoryLinkBindMounted(ltr)
	case LocalVolumeTmpfs:
		l.cleanupLocalVolumeTmpfs(ltr)
	case LocalVolumeBlock:
		l.cleanupLocalVolumeBlock(ltr)
	case LocalVolumeBlockFS:
		l.cleanupLocalVolumeBlockFS(ltr)
	case LocalVolumeGCELocalSSD:
		l.cleanupLocalVolumeGCELocalSSD(ltr)
	default:
		framework.Failf("Failed to remove local test resource, unsupported volume type: %v is specified", ltr.VolumeType)
	}
	return
}
