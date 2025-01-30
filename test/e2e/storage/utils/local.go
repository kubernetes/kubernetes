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
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
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
	Create(ctx context.Context, node *v1.Node, volumeType LocalVolumeType, parameters map[string]string) *LocalTestResource
	ExpandBlockDevice(ctx context.Context, ltr *LocalTestResource, mbToAdd int) error
	Remove(ctx context.Context, ltr *LocalTestResource)
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

func (l *ltrMgr) setupLocalVolumeTmpfs(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	ginkgo.By(fmt.Sprintf("Creating tmpfs mount point on node %q at path %q", node.Name, hostDir))
	err := l.hostExec.IssueCommand(ctx, fmt.Sprintf("mkdir -p %q && mount -t tmpfs -o size=10m tmpfs-%q %q", hostDir, hostDir, hostDir), node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeTmpfs(ctx context.Context, ltr *LocalTestResource) {
	ginkgo.By(fmt.Sprintf("Unmount tmpfs mount point on node %q at path %q", ltr.Node.Name, ltr.Path))
	err := l.hostExec.IssueCommand(ctx, fmt.Sprintf("umount %q", ltr.Path), ltr.Node)
	framework.ExpectNoError(err)

	ginkgo.By("Removing the test directory")
	err = l.hostExec.IssueCommand(ctx, fmt.Sprintf("rm -r %s", ltr.Path), ltr.Node)
	framework.ExpectNoError(err)
}

// createAndSetupLoopDevice creates an empty file and associates a loop devie with it.
func (l *ltrMgr) createAndSetupLoopDevice(ctx context.Context, dir string, node *v1.Node, size int) {
	ginkgo.By(fmt.Sprintf("Creating block device on node %q using path %q", node.Name, dir))
	mkdirCmd := fmt.Sprintf("mkdir -p %s", dir)
	count := size / 4096
	// xfs requires at least 4096 blocks
	if count < 4096 {
		count = 4096
	}
	ddCmd := fmt.Sprintf("dd if=/dev/zero of=%s/file bs=4096 count=%d", dir, count)
	losetupCmd := fmt.Sprintf("losetup -f %s/file", dir)
	err := l.hostExec.IssueCommand(ctx, fmt.Sprintf("%s && %s && %s", mkdirCmd, ddCmd, losetupCmd), node)
	framework.ExpectNoError(err)
}

// findLoopDevice finds loop device path by its associated storage directory.
func (l *ltrMgr) findLoopDevice(ctx context.Context, dir string, node *v1.Node) string {
	cmd := fmt.Sprintf("E2E_LOOP_DEV=$(losetup | grep %s/file | awk '{ print $1 }') 2>&1 > /dev/null && echo ${E2E_LOOP_DEV}", dir)
	loopDevResult, err := l.hostExec.IssueCommandWithResult(ctx, cmd, node)
	framework.ExpectNoError(err)
	return strings.TrimSpace(loopDevResult)
}

func (l *ltrMgr) setupLocalVolumeBlock(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	loopDir := l.getTestDir()
	l.createAndSetupLoopDevice(ctx, loopDir, node, 20*1024*1024)
	loopDev := l.findLoopDevice(ctx, loopDir, node)
	return &LocalTestResource{
		Node:    node,
		Path:    loopDev,
		loopDir: loopDir,
	}
}

// teardownLoopDevice tears down loop device by its associated storage directory.
func (l *ltrMgr) teardownLoopDevice(ctx context.Context, dir string, node *v1.Node) {
	loopDev := l.findLoopDevice(ctx, dir, node)
	ginkgo.By(fmt.Sprintf("Tear down block device %q on node %q at path %s/file", loopDev, node.Name, dir))
	losetupDeleteCmd := fmt.Sprintf("losetup -d %s", loopDev)
	err := l.hostExec.IssueCommand(ctx, losetupDeleteCmd, node)
	framework.ExpectNoError(err)
	return
}

func (l *ltrMgr) cleanupLocalVolumeBlock(ctx context.Context, ltr *LocalTestResource) {
	l.teardownLoopDevice(ctx, ltr.loopDir, ltr.Node)
	ginkgo.By(fmt.Sprintf("Removing the test directory %s", ltr.loopDir))
	removeCmd := fmt.Sprintf("rm -r %s", ltr.loopDir)
	err := l.hostExec.IssueCommand(ctx, removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeBlockFS(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	ltr := l.setupLocalVolumeBlock(ctx, node, parameters)
	loopDev := ltr.Path
	loopDir := ltr.loopDir
	// Format and mount at loopDir and give others rwx for read/write testing
	cmd := fmt.Sprintf("mkfs -t ext4 %s && mount -t ext4 %s %s && chmod o+rwx %s", loopDev, loopDev, loopDir, loopDir)
	err := l.hostExec.IssueCommand(ctx, cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node:    node,
		Path:    loopDir,
		loopDir: loopDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeBlockFS(ctx context.Context, ltr *LocalTestResource) {
	umountCmd := fmt.Sprintf("umount %s", ltr.Path)
	err := l.hostExec.IssueCommand(ctx, umountCmd, ltr.Node)
	framework.ExpectNoError(err)
	l.cleanupLocalVolumeBlock(ctx, ltr)
}

func (l *ltrMgr) setupLocalVolumeDirectory(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	mkdirCmd := fmt.Sprintf("mkdir -p %s", hostDir)
	err := l.hostExec.IssueCommand(ctx, mkdirCmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectory(ctx context.Context, ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", ltr.Path)
	err := l.hostExec.IssueCommand(ctx, removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeDirectoryLink(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	hostDirBackend := hostDir + "-backend"
	cmd := fmt.Sprintf("mkdir -p %s && ln -s %s %s", hostDirBackend, hostDirBackend, hostDir)
	err := l.hostExec.IssueCommand(ctx, cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryLink(ctx context.Context, ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	hostDir := ltr.Path
	hostDirBackend := hostDir + "-backend"
	removeCmd := fmt.Sprintf("rm -r %s && rm -r %s", hostDir, hostDirBackend)
	err := l.hostExec.IssueCommand(ctx, removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeDirectoryBindMounted(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	cmd := fmt.Sprintf("mkdir -p %s && mount --bind %s %s", hostDir, hostDir, hostDir)
	err := l.hostExec.IssueCommand(ctx, cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryBindMounted(ctx context.Context, ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	hostDir := ltr.Path
	removeCmd := fmt.Sprintf("umount %s && rm -r %s", hostDir, hostDir)
	err := l.hostExec.IssueCommand(ctx, removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeDirectoryLinkBindMounted(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	hostDir := l.getTestDir()
	hostDirBackend := hostDir + "-backend"
	cmd := fmt.Sprintf("mkdir -p %s && mount --bind %s %s && ln -s %s %s", hostDirBackend, hostDirBackend, hostDirBackend, hostDirBackend, hostDir)
	err := l.hostExec.IssueCommand(ctx, cmd, node)
	framework.ExpectNoError(err)
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeDirectoryLinkBindMounted(ctx context.Context, ltr *LocalTestResource) {
	ginkgo.By("Removing the test directory")
	hostDir := ltr.Path
	hostDirBackend := hostDir + "-backend"
	removeCmd := fmt.Sprintf("rm %s && umount %s && rm -r %s", hostDir, hostDirBackend, hostDirBackend)
	err := l.hostExec.IssueCommand(ctx, removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) setupLocalVolumeGCELocalSSD(ctx context.Context, node *v1.Node, parameters map[string]string) *LocalTestResource {
	res, err := l.hostExec.IssueCommandWithResult(ctx, "ls /mnt/disks/by-uuid/google-local-ssds-scsi-fs/", node)
	framework.ExpectNoError(err)
	dirName := strings.Fields(res)[0]
	hostDir := "/mnt/disks/by-uuid/google-local-ssds-scsi-fs/" + dirName
	return &LocalTestResource{
		Node: node,
		Path: hostDir,
	}
}

func (l *ltrMgr) cleanupLocalVolumeGCELocalSSD(ctx context.Context, ltr *LocalTestResource) {
	// This filesystem is attached in cluster initialization, we clean all files to make it reusable.
	removeCmd := fmt.Sprintf("find '%s' -mindepth 1 -maxdepth 1 -print0 | xargs -r -0 rm -rf", ltr.Path)
	err := l.hostExec.IssueCommand(ctx, removeCmd, ltr.Node)
	framework.ExpectNoError(err)
}

func (l *ltrMgr) expandLocalVolumeBlockFS(ctx context.Context, ltr *LocalTestResource, mbToAdd int) error {
	ddCmd := fmt.Sprintf("dd if=/dev/zero of=%s/file conv=notrunc oflag=append bs=1M count=%d", ltr.loopDir, mbToAdd)
	loopDev := l.findLoopDevice(ctx, ltr.loopDir, ltr.Node)
	losetupCmd := fmt.Sprintf("losetup -c %s", loopDev)
	return l.hostExec.IssueCommand(ctx, fmt.Sprintf("%s && %s", ddCmd, losetupCmd), ltr.Node)
}

func (l *ltrMgr) ExpandBlockDevice(ctx context.Context, ltr *LocalTestResource, mbtoAdd int) error {
	switch ltr.VolumeType {
	case LocalVolumeBlockFS:
		return l.expandLocalVolumeBlockFS(ctx, ltr, mbtoAdd)
	}
	return fmt.Errorf("Failed to expand local test resource, unsupported volume type: %s", ltr.VolumeType)
}

func (l *ltrMgr) Create(ctx context.Context, node *v1.Node, volumeType LocalVolumeType, parameters map[string]string) *LocalTestResource {
	var ltr *LocalTestResource
	switch volumeType {
	case LocalVolumeDirectory:
		ltr = l.setupLocalVolumeDirectory(ctx, node, parameters)
	case LocalVolumeDirectoryLink:
		ltr = l.setupLocalVolumeDirectoryLink(ctx, node, parameters)
	case LocalVolumeDirectoryBindMounted:
		ltr = l.setupLocalVolumeDirectoryBindMounted(ctx, node, parameters)
	case LocalVolumeDirectoryLinkBindMounted:
		ltr = l.setupLocalVolumeDirectoryLinkBindMounted(ctx, node, parameters)
	case LocalVolumeTmpfs:
		ltr = l.setupLocalVolumeTmpfs(ctx, node, parameters)
	case LocalVolumeBlock:
		ltr = l.setupLocalVolumeBlock(ctx, node, parameters)
	case LocalVolumeBlockFS:
		ltr = l.setupLocalVolumeBlockFS(ctx, node, parameters)
	case LocalVolumeGCELocalSSD:
		ltr = l.setupLocalVolumeGCELocalSSD(ctx, node, parameters)
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

func (l *ltrMgr) Remove(ctx context.Context, ltr *LocalTestResource) {
	switch ltr.VolumeType {
	case LocalVolumeDirectory:
		l.cleanupLocalVolumeDirectory(ctx, ltr)
	case LocalVolumeDirectoryLink:
		l.cleanupLocalVolumeDirectoryLink(ctx, ltr)
	case LocalVolumeDirectoryBindMounted:
		l.cleanupLocalVolumeDirectoryBindMounted(ctx, ltr)
	case LocalVolumeDirectoryLinkBindMounted:
		l.cleanupLocalVolumeDirectoryLinkBindMounted(ctx, ltr)
	case LocalVolumeTmpfs:
		l.cleanupLocalVolumeTmpfs(ctx, ltr)
	case LocalVolumeBlock:
		l.cleanupLocalVolumeBlock(ctx, ltr)
	case LocalVolumeBlockFS:
		l.cleanupLocalVolumeBlockFS(ctx, ltr)
	case LocalVolumeGCELocalSSD:
		l.cleanupLocalVolumeGCELocalSSD(ctx, ltr)
	default:
		framework.Failf("Failed to remove local test resource, unsupported volume type: %v is specified", ltr.VolumeType)
	}
	return
}
