// Copyright 2016 The rkt Authors
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

package pod

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"strconv"

	"github.com/appc/spec/schema"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/lock"
	"github.com/hashicorp/errwrap"
)

// ErrImmutable is the error that is returned by SandboxManifest if the pod is immutable,
// hence changes on the pod manifest are not allowed.
var ErrImmutable = errors.New("immutable pod")

// SandboxManifest loads the underlying pod manifest and checks whether mutable operations are allowed.
// It returns ErrImmutable if the pod does not allow mutable operations or any other error if the operation failed.
// Upon success a reference to the pod manifest is returned and mutable operations are possible.
func (p *Pod) SandboxManifest() (*schema.PodManifest, error) {
	_, pm, err := p.PodManifest() // this takes the lock fd to load the manifest, hence path is not needed here
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error loading pod manifest"), err)
	}

	ms, ok := pm.Annotations.Get("coreos.com/rkt/stage1/mutable")
	if ok {
		p.mutable, err = strconv.ParseBool(ms)
		if err != nil {
			return nil, errwrap.Wrap(errors.New("error parsing mutable annotation"), err)
		}
	}

	if !p.mutable {
		return nil, ErrImmutable
	}

	return pm, nil
}

// UpdateManifest updates the given pod manifest in the given path atomically on the file system.
// The pod manifest has to be locked using LockManifest first to avoid races in case of concurrent writes.
func (p *Pod) UpdateManifest(m *schema.PodManifest, path string) error {
	if !p.mutable {
		return ErrImmutable
	}

	mpath := common.PodManifestPath(path)
	mstat, err := os.Stat(mpath)
	if err != nil {
		return err
	}

	tmpf, err := ioutil.TempFile(path, "")
	if err != nil {
		return err
	}

	defer func() {
		tmpf.Close()
		os.Remove(tmpf.Name())
	}()

	if err := tmpf.Chmod(mstat.Mode().Perm()); err != nil {
		return err
	}

	if err := json.NewEncoder(tmpf).Encode(m); err != nil {
		return err
	}

	if err := os.Rename(tmpf.Name(), mpath); err != nil {
		return err
	}

	return nil
}

// ExclusiveLockManifest gets an exclusive lock on only the pod manifest in the app sandbox.
// Since the pod is already running, we won't be able to get an exclusive lock on the pod itself.
func (p *Pod) ExclusiveLockManifest() error {
	if !p.isRunning() {
		return errors.New("pod is not running")
	}

	if p.manifestLock != nil {
		return p.manifestLock.ExclusiveLock() // This is idempotent
	}

	l, err := lock.ExclusiveLock(common.PodManifestLockPath(p.Path()), lock.RegFile)
	if err != nil {
		return err
	}

	p.manifestLock = l
	return nil
}

// UnlockManifest unlocks the pod manifest lock.
func (p *Pod) UnlockManifest() error {
	if !p.isRunning() {
		return errors.New("pod is not running")
	}

	if p.manifestLock == nil {
		return nil
	}

	if err := p.manifestLock.Close(); err != nil {
		return err
	}
	p.manifestLock = nil
	return nil
}
