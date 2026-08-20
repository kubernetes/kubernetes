/*
Copyright 2020 The Kubernetes Authors.

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

package etcd

import (
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// CreateDataDirectory creates the etcd data directory (commonly /var/lib/etcd) with the right permissions.
func CreateDataDirectory(dir string) error {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return errors.Wrapf(err, "failed to create the etcd data directory: %q", dir)
	}
	return nil
}

// BackupDataDirectory copies a live etcd data directory for upgrade rollback.
//
// The destination layout matches `cp -r src dst` when dst already exists as a
// directory: contents are placed under dst/basename(src).
//
// Unlike a plain recursive cp, this:
//   - skips ephemeral WAL pre-allocation files (*.tmp), which etcd's
//     filePipeline renames into place during segment rotation
//   - ignores files that disappear mid-copy (os.ErrNotExist), which can happen
//     when those *.tmp files are renamed, or when old WAL segments are removed
//
// etcd ignores *.tmp when reading a WAL directory, so omitting them does not
// affect restore usability of the backup.
func BackupDataDirectory(src, dst string) error {
	srcInfo, err := os.Stat(src)
	if err != nil {
		return errors.Wrapf(err, "failed to stat etcd data directory %q", src)
	}
	if !srcInfo.IsDir() {
		return errors.Errorf("etcd data directory %q is not a directory", src)
	}

	// Match `cp -r src dst` when dst exists as a directory.
	target := filepath.Join(dst, filepath.Base(src))
	if err := os.MkdirAll(target, srcInfo.Mode()); err != nil {
		return errors.Wrapf(err, "failed to create etcd backup directory %q", target)
	}

	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				klog.V(4).Infof("skipping vanished path during etcd backup: %s", path)
				return nil
			}
			return err
		}
		if path == src {
			return nil
		}

		// Ephemeral WAL pre-allocation files are renamed by etcd and must not
		// fail the backup. etcd itself ignores leftover *.tmp on restore.
		if strings.HasSuffix(d.Name(), ".tmp") {
			klog.V(4).Infof("skipping ephemeral file during etcd backup: %s", path)
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}

		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		destPath := filepath.Join(target, rel)

		info, err := d.Info()
		if err != nil {
			if os.IsNotExist(err) {
				klog.V(4).Infof("skipping vanished path during etcd backup: %s", path)
				return nil
			}
			return err
		}

		if info.IsDir() {
			if err := os.MkdirAll(destPath, info.Mode()); err != nil {
				return errors.Wrapf(err, "failed to create directory %q", destPath)
			}
			return nil
		}

		if !info.Mode().IsRegular() {
			klog.V(4).Infof("skipping non-regular file during etcd backup: %s", path)
			return nil
		}

		if err := copyFile(path, destPath, info.Mode()); err != nil {
			if os.IsNotExist(err) {
				klog.V(4).Infof("skipping vanished file during etcd backup: %s", path)
				return nil
			}
			return errors.Wrapf(err, "failed to copy %q to %q", path, destPath)
		}
		return nil
	})
}

func copyFile(src, dst string, mode fs.FileMode) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() {
		_ = sourceFile.Close()
	}()

	if err := os.MkdirAll(filepath.Dir(dst), 0700); err != nil {
		return err
	}

	destFile, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	defer func() {
		_ = destFile.Close()
	}()

	_, err = io.Copy(destFile, sourceFile)
	return err
}
