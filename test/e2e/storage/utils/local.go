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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
)

// LocalVolumeType represents type of local volume, e.g. tmpfs, directory,
// block, etc.
type LocalVolumeType string

const (
	// A simple directory as local volume
	LocalVolumeDirectory LocalVolumeType = "dir"
	// Like LocalVolumeDirectory but it's a symbolic link to directory
	LocalVolumeDirectoryLink LocalVolumeType = "dir-link"
	// Like LocalVolumeDirectory but bind mounted
	LocalVolumeDirectoryBindMounted LocalVolumeType = "dir-bindmounted"
	// Like LocalVolumeDirectory but it's a symbolic link to self bind mounted directory
	// Note that bind mounting at symbolic link actually mounts at directory it
	// links to
	LocalVolumeDirectoryLinkBindMounted LocalVolumeType = "dir-link-bindmounted"
	// Use temporary filesystem as local volume
	LocalVolumeTmpfs LocalVolumeType = "tmpfs"
	// Block device, creates a local file, and maps it as a block device
	LocalVolumeBlock LocalVolumeType = "block"
	// Filesystem backed by a block device
	LocalVolumeBlockFS LocalVolumeType = "blockfs"
	// Use GCE Local SSD as local volume, this is a filesystem
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
	By(fmt.Sprintf("Creating tmpfs mount point on node %q at path %q", node.Name, hostDir))
	err := l.hostExec.IssueCommand(fmt.Sprintf("mkdir -p %q && sudo mount -t tmpfs -o size=10m tmpfs-%q %q", hostDir, hostDir, hostDir), node)
	Expect(err).NotTo(HaveOccurred())
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeTmpfs(ltr *LocalTestResource) {
	By(fmt.Sprintf("Unmount tmpfs mount point on node %q at path %q", ltr.Node.Name, ltr.Path))
	err := l.hostExec.IssueCommand(fmt.Sprintf("sudo umount %q", ltr.Path), ltr.Node)
	Expect(err).NotTo(HaveOccurred())

	By("Removing the test directory")
	err = l.hostExec.IssueCommand(fmt.Sprintf("rm -r %s", ltr.Path), ltr.Node)
	Expect(err).NotTo(HaveOccurred())
}

// createAndSetupLoopDevice creates an empty file and associates a loop devie with it.
func (l *ltrMgr) createAndSetupLoopDevice(dir string, node *v1.Node, size int) {
	By(fmt.Sprintf("Creating block device on node %q using path %q", node.Name, dir))
	mkdirCmd := fmt.Sprintf("mkdir -p %s", dir)
	count := size / 4096
	// xfs requires at least 4096 blocks
	if count < 4096 {
		count = 4096
	}
	ddCmd := fmt.Sprintf("dd if=/dev/zero of=%s/file bs=4096 count=%d", dir, count)
	losetupCmd := fmt.Sprintf("sudo losetup -f %s/file", dir)
	err := l.hostExec.IssueCommand(fmt.Sprintf("%s && %s && %s", mkdirCmd, ddCmd, losetupCmd), node)
	Expect(err).NotTo(HaveOccurred())
}

// findLoopDevice finds loop device path by its associated storage directory.
func (l *ltrMgr) findLoopDevice(dir string, node *v1.Node) string {
	cmd := fmt.Sprintf("E2E_LOOP_DEV=$(sudo losetup | grep %s/file | awk '{ print $1 }') 2>&1 > /dev/null && echo ${E2E_LOOP_DEV}", dir)
	loopDevResult, err := l.hostExec.IssueCommandWithResult(cmd, node)
	Expect(err).NotTo(HaveOccurred())
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
	By(fmt.Sprintf("Tear down block device %q on node %q at path %s/file", loopDev, node.Name, dir))
	losetupDeleteCmd := fmt.Sprintf("sudo losetup -d %s", loopDev)
	err := l.hostExec.IssueCommand(losetupDeleteCmd, node)
	Expect(err).NotTo(HaveOccurred())
	return
}

func (l *ltrMgr) cleanupLocalVolumeBlock(ltr *LocalTestResource) {
	l.teardownLoopDevice(ltr.loopDir, ltr.Node)
	By(fmt.Sprintf("Removing the test directory %s", ltr.loopDir))
	removeCmd := fmt.Sprintf("rm -r %s", ltr.loopDir)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	Expect(err).NotTo(HaveOccurred())
}

func (l *ltrMgr) setupLocalVolumeBlockFS(node *v1.Node, parameters map[string]string) *LocalTestResource {
	ltr := l.setupLocalVolumeBlock(node, parameters)
	loopDev := ltr.Path
	loopDir := ltr.loopDir
	// Format and mount at loopDir and give others rwx for read/write testing
	cmd := fmt.Sprintf("sudo mkfs -t ext4 %s && sudo mount -t ext4 %s %s && sudo chmod o+rwx %s", loopDev, loopDev, loopDir, loopDir)
	err := l.hostExec.IssueCommand(cmd, node)
	Expect(err).NotTo(HaveOccurred())
	return &LocalTestResource{
		Node:    node,
		Path:    loopDir,
		loopDir: loopDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeBlockFS(ltr *LocalTestResource) {
	umountCmd := fmt.Sprintf("sudo umount %s", ltr.Path)
	err := l.hostExec.IssueCommand(umountCmd, ltr.Node)
	Expect(err).NotTo(HaveOccurred())
	l.cleanupLocalVolumeBlock(ltr)
}

func (l *ltrMgr) setupLocalVolumeDirectory(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	mkdirCmd := fmt.Sprintf("mkdir -p %s", hostDir)
	err := l.hostExec.IssueCommand(mkdirCmd, node)
	Expect(err).NotTo(HaveOccurred())
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectory(ltr *LocalTestResource) {
	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", ltr.Path)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	Expect(err).NotTo(HaveOccurred())
}

func (l *ltrMgr) setupLocalVolumeDirectoryLink(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	hostDirBackend := hostDir + "-backend"
	cmd := fmt.Sprintf("mkdir %s && sudo ln -s %s %s", hostDirBackend, hostDirBackend, hostDir)
	err := l.hostExec.IssueCommand(cmd, node)
	Expect(err).NotTo(HaveOccurred())
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryLink(ltr *LocalTestResource) {
	By("Removing the test directory")
	hostDir := ltr.Path
	hostDirBackend := hostDir + "-backend"
	removeCmd := fmt.Sprintf("sudo rm -r %s && rm -r %s", hostDir, hostDirBackend)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	Expect(err).NotTo(HaveOccurred())
}

func (l *ltrMgr) setupLocalVolumeDirectoryBindMounted(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	cmd := fmt.Sprintf("mkdir %s && sudo mount --bind %s %s", hostDir, hostDir, hostDir)
	err := l.hostExec.IssueCommand(cmd, node)
	Expect(err).NotTo(HaveOccurred())
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryBindMounted(ltr *LocalTestResource) {
	By("Removing the test directory")
	hostDir := ltr.Path
	removeCmd := fmt.Sprintf("sudo umount %s && rm -r %s", hostDir, hostDir)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	Expect(err).NotTo(HaveOccurred())
}

func (l *ltrMgr) setupLocalVolumeDirectoryLinkBindMounted(node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	hostDirBackend := hostDir + "-backend"
	cmd := fmt.Sprintf("mkdir %s && sudo mount --bind %s %s && sudo ln -s %s %s", hostDirBackend, hostDirBackend, hostDirBackend, hostDirBackend, hostDir)
	err := l.hostExec.IssueCommand(cmd, node)
	Expect(err).NotTo(HaveOccurred())
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryLinkBindMounted(ltr *LocalTestResource) {
	By("Removing the test directory")
	hostDir := ltr.Path
	hostDirBackend := hostDir + "-backend"
	removeCmd := fmt.Sprintf("sudo rm %s && sudo umount %s && rm -r %s", hostDir, hostDirBackend, hostDirBackend)
	err := l.hostExec.IssueCommand(removeCmd, ltr.Node)
	Expect(err).NotTo(HaveOccurred())
}

func (l *ltrMgr) setupLocalVolumeGCELocalSSD(node *v1.Node, parameters map[string]string) *LocalTestResource {
	res, err := l.hostExec.IssueCommandWithResult("ls /mnt/disks/by-uuid/google-local-ssds-scsi-fs/", node)
	Expect(err).NotTo(HaveOccurred())
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
	Expect(err).NotTo(HaveOccurred())
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
