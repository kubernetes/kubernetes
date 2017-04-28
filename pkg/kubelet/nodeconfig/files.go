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

package nodeconfig

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	checkpointsDir = "checkpoints"
	initConfigDir  = "init"

	curSymlink     = ".cur"
	lkgSymlink     = ".lkg"
	tmpSymlink     = ".tmp"
	defaultSymlink = "/dev/zero"

	defaultPerm = 0666
)

// ensureConfigDir makes sure that the node-config-dir is setup properly.
// If something prevents correct setup, a fatal error occurs.
func (cc *NodeConfigController) ensureCfgDir() {
	const errfmt = "failed to ensure node-config-dir %q is set up properly, error: %v"
	infof("ensuring node-confg-dir %q is set up correctly", cc.configDir)
	// node-config-dir
	if err := cc.ensureDir(""); err != nil {
		fatalf(errfmt, cc.configDir, err)
	}
	// checkpoints dir
	if err := cc.ensureDir(checkpointsDir); err != nil {
		fatalf(errfmt, cc.configDir, err)
	}
	// symlink to current config checkpoint
	if err := cc.ensureSymlink(curSymlink); err != nil {
		fatalf(errfmt, cc.configDir, err)
	}
	// symlink to last-known-good config checkpoint
	if err := cc.ensureSymlink(lkgSymlink); err != nil {
		fatalf(errfmt, cc.configDir, err)
	}
	// bad config tracking file
	if err := cc.ensureFile(badConfigsFile); err != nil {
		fatalf(errfmt, cc.configDir, err)
	}
	// startups tracking file
	if err := cc.ensureFile(startupsFile); err != nil {
		fatalf(errfmt, cc.configDir, err)
	}
}

// curModTime returns the modification time the current config.
// If the modification time cannot be determined, a fatal error occurs.
func (cc *NodeConfigController) curModTime() time.Time {
	path := filepath.Join(cc.configDir, curSymlink)
	info, err := os.Lstat(path)
	if err != nil {
		fatalf("failed to lstat %q, error: %v", path, err)
	}
	stamp := info.ModTime()
	return stamp
}

// curIsLkg returns true if curSymlink and lkgSymlink point to the same place, false otherwise.
// If filesystem issues prevent reading the link, a fatal error occurs.
func (cc *NodeConfigController) curIsLkg() bool {
	curp := filepath.Join(cc.configDir, curSymlink)
	cur, err := os.Readlink(curp)
	if err != nil {
		fatalf("failed to read link %q, error:", curp, err)
	}

	lkgp := filepath.Join(cc.configDir, lkgSymlink)
	lkg, err := os.Readlink(lkgp)
	if err != nil {
		fatalf("failed to read link %q, error:", lkgp, err)
	}

	return cur == lkg
}

// setCurAsLkg replaces lkgSymlink so it points to the same location as `curSymlink`.
// If filesystem issues prevent these operations, a fatal error occurs.
// This is used for updating the LKG when a config exits its trial period.
func (cc *NodeConfigController) setCurAsLkg() {
	path := filepath.Join(cc.configDir, curSymlink)
	dest, err := os.Readlink(path)
	if err != nil {
		fatalf("failed to read link %q, error: %v", path, err)
	}
	cc.setSymlink(lkgSymlink, dest)
}

// curInTrial returns true if the time elapsed since the last modification to `curSymlink`
// exceeds `trialDur`, false otherwise.
// If filesystem issues prevent determining the modification time of `curSymlink`, a fatal error occurs.
func (cc *NodeConfigController) curInTrial(trialDur time.Duration) bool {
	now := time.Now()
	t := cc.curModTime()
	if now.Sub(t) > trialDur {
		return true
	}
	return false
}

// curUID returns the uid of the current config, or the empty string if the current config is the default.
// If the current config cannot be determined, a fatal error occurs.
func (cc *NodeConfigController) curUID() string {
	uid, err := cc.symlinkUID(curSymlink)
	if err != nil {
		fatalf("failed to determine the current configuration, error: %v", err)
	}
	return uid
}

// curUID returns the uid of the last-known-good config, or the empty string if the last-known-good config is the default.
// If the last-known-good config cannot be determined, a fatal error occurs.
func (cc *NodeConfigController) lkgUID() string {
	uid, err := cc.symlinkUID(lkgSymlink)
	if err != nil {
		fatalf("failed to determine the last-known-good configuration, error: %v", err)
	}
	return uid
}

// symlinkUID returns the UID of the config the symlink at `cc.configDir/relPath` points to,
// or the empty string if it points to the default location.
// If a filesystem error occurs, the error is returned.
func (cc *NodeConfigController) symlinkUID(relPath string) (string, error) {
	path := filepath.Join(cc.configDir, relPath)
	dest, err := os.Readlink(path)
	if err != nil {
		return "", err
	} else if dest == defaultSymlink {
		return "", nil
	}
	return filepath.Base(dest), nil
}

// setSymlinkUID points the symlink at `cc.configDir/relPath` to the checkpoint directory for `uid`.
// If filesystem issues prevent the symlink being set, a fatal error occurs.
func (cc *NodeConfigController) setSymlinkUID(relPath string, uid string) {
	cc.setSymlink(relPath, filepath.Join(cc.configDir, checkpointsDir, uid))
}

// resetSymlink points the symlink at `cc.configDir/relPath` to the default location.
// If the symlink was reset, returns true. Otherwise returns false.
// If filesystem issues prevent the symlink being reset, a fatal error occurs.
func (cc *NodeConfigController) resetSymlink(relPath string) bool {
	path := filepath.Join(cc.configDir, relPath)
	ln, err := os.Readlink(path)
	if err != nil {
		fatalf("failed to read link %q, error: %v", path, err)
	}
	// don't need to reset if it already points to the default location
	if ln == defaultSymlink {
		return false
	}
	cc.setSymlink(relPath, defaultSymlink)
	return true
}

// setSymlink points the symlink at `cc.configDir/relPath` to `dest`.
// If the symlink was set, returns true. Otherwise returns false.
// If the symlink does not already exist, a fatal error occurs. Use ensureSymlink to create it before using setSymlink.
// If filesystem issues prevent the symlink from being set, a fatal error occurs.
func (cc *NodeConfigController) setSymlink(relPath string, dest string) {
	path := filepath.Join(cc.configDir, relPath)
	tmpPath := filepath.Join(cc.configDir, tmpSymlink)

	// require that symlink exist, as ensureSymlink should be used to create it
	if ok, err := cc.symlinkExists(relPath); err != nil {
		fatalf("failed checking whether symlink exists at %q, error: %v", path, err)
	} else if !ok {
		fatalf("symlink must already exist to set symlink, target path: %q, error: %v", path, err)
	} // Assert: symlink exists

	// delete the temporary symlink if it exists, in most cases it will not, but the
	// Kubelet could have crashed between creating and renaming the temporary symlink
	if ok, err := cc.symlinkExists(tmpPath); err != nil {
		fatalf("failed checking whether temporary symlink exists at %q, error: %v", tmpPath, err)
	} else if ok {
		// Assert: temporary symlink exists, we must delete it
		if err := os.Remove(tmpPath); err != nil {
			fatalf("failed to remove temporary symlink %q, error: %v", tmpPath, err)
		}
	}

	// create the temporary symlink, and then rename it to atomically set the target symlink
	if err := os.Symlink(dest, tmpPath); err != nil {
		fatalf("failed to create temporary symlink %q, error: %v", tmpPath, err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		fatalf("failed to rename temporary symlink %q to %q (attempting to set symlink at the latter path), error: %v", tmpPath, path, err)
	}
}

// symlinkExists returns (true, nil) if `cc.configDir/relPath` is a symlink and we are able lstat it.
// If `relPath` exists but is not a symlink, returns an error.
// If `relPath` does not exist, returns (false, nil).
// If lstat fails, returns an error.
func (cc *NodeConfigController) symlinkExists(relPath string) (bool, error) {
	path := filepath.Join(cc.configDir, relPath)
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return true, nil
		}
		return false, fmt.Errorf("expected symlink at %q, but mode is %q", path, info.Mode().String())
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// ensureSymlink ensures that a symlink exists at `cc.configDir/relPath`.
// If the symlink does not exist, it is created to point at the default location.
// If filesystem issues prevent ensuring the symlink, returns an error.
func (cc *NodeConfigController) ensureSymlink(relPath string) error {
	// if symlink exists, don't change it, but do report any unexpected errors
	if ok, err := cc.symlinkExists(relPath); ok || err != nil {
		return err // if ok, err is nil
	} // Assert: symlink does not exist

	// create the symlink
	path := filepath.Join(cc.configDir, relPath)
	if err := os.Symlink(defaultSymlink, path); err != nil {
		return err
	}
	return nil
}

// fileExists returns (true, nil) if `cc.configDir/relPath` is a regular file and we are able stat it.
// If `relPath` exists but is not a regular file, returns an error.
// If `relPath` does not exist, returns (false, nil).
// If stat fails, returns an error.
func (cc *NodeConfigController) fileExists(relPath string) (bool, error) {
	path := filepath.Join(cc.configDir, relPath)
	if info, err := os.Stat(path); err == nil {
		if info.Mode().IsRegular() {
			return true, nil
		}
		return false, fmt.Errorf("expected regular file at %q, but mode is %q", path, info.Mode().String())
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// ensureFile ensures that a regular file exists at `cc.configDir/relPath`.
// If the file does not exist, an empty regular file is created.
// If filesystem issues prevent ensuring the file, returns an error.
func (cc *NodeConfigController) ensureFile(relPath string) error {
	// if file exists, don't change it, but do report any unexpected errors
	if ok, err := cc.fileExists(relPath); ok || err != nil {
		return err
	} // Assert: file does not exist

	// create the file
	path := filepath.Join(cc.configDir, relPath)
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	// close the file, since we don't intend to use it yet
	if err := file.Close(); err != nil {
		return err
	}
	return nil
}

// dirExists returns (true, nil) if `cc.configDir/relPath` is a directory and we are able stat it.
// If `relPath` exists but is not a directory, returns an error.
// If `relPath` does not exist, returns (false, nil).
// If stat fails, returns an error.
func (cc *NodeConfigController) dirExists(relPath string) (bool, error) {
	path := filepath.Join(cc.configDir, relPath)
	if info, err := os.Stat(path); err == nil {
		if info.IsDir() {
			return true, nil
		}
		return false, fmt.Errorf("expected dir at %q, but mode is is %q", path, info.Mode().String())
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// ensureDir ensures that a directory exists at `cc.configDir/relPath`.
// If the directory does not exist, an empty directory is created.
// If filesystem issues prevent ensuring the directory, returns an error.
func (cc *NodeConfigController) ensureDir(relPath string) error {
	// if dir exists, don't change it, but do report any unexpected errors
	if ok, err := cc.dirExists(relPath); ok || err != nil {
		return err
	} // Assert: dir does not exist

	// create the dir
	path := filepath.Join(cc.configDir, relPath)
	if err := os.Mkdir(path, defaultPerm); err != nil {
		return err
	}
	return nil
}
