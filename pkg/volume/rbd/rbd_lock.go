/*
Copyright 2016 The Kubernetes Authors.

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

//
// utility functions to setup rbd volume
// mainly implement diskManager interface
//

package rbd

import (
	"math/rand"
	"strings"

	"github.com/golang/glog"
)

// Abstract interface to RBD locker.
type Locker interface {
	// lock rbd image for a host
	Fencing(b rbdMounter, hostName string) error
	// release lock for a host
	Defencing(c rbdMounter, hostName string) error
	// query lock status
	IsLocked(c rbdMounter, hostName string) (bool, error)
}

type RBDLocker struct{}

func (locker *RBDLocker) rbdLock(b rbdMounter, hostName string, lock bool, queryOnly bool) (bool, error) {
	var err error
	var output, imageLocker string
	var cmd []byte
	var secret_opt []string

	if b.Secret != "" {
		secret_opt = []string{"--key=" + b.Secret}
	} else {
		secret_opt = []string{"-k", b.Keyring}
	}
	// construct lock id using host name and a magic prefix
	lock_id := "kubelet_lock_magic_" + hostName

	l := len(b.Mon)
	// avoid mount storm, pick a host randomly
	start := rand.Int() % l
	// iterate all hosts until mount succeeds.
	for i := start; i < start+l; i++ {
		mon := b.Mon[i%l]
		// cmd "rbd lock list" serves two purposes:
		// for fencing, check if lock already held for this host
		// this edge case happens if host crashes in the middle of acquiring lock and mounting rbd
		// for defencing, get the locker name, something like "client.1234"
		cmd, err = b.plugin.execCommand("rbd",
			append([]string{"lock", "list", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon}, secret_opt...))
		output = string(cmd)

		if err != nil {
			glog.Warningf("failed to list rbd image %v/%v, output: %v, err: %v", b.Pool, b.Image, output, err)
			continue
		}

		if lock {
			// check if lock is already held for this host by matching lock_id and rbd lock id
			if strings.Contains(output, lock_id) {
				// this host already holds the lock, exit
				glog.V(1).Infof("rbd: lock already held for %s", lock_id)
				return true, nil
			}
			if queryOnly {
				return false, nil
			}
			// hold a lock: rbd lock add
			cmd, err = b.plugin.execCommand("rbd",
				append([]string{"lock", "add", b.Image, lock_id, "--pool", b.Pool, "--id", b.Id, "-m", mon}, secret_opt...))
		} else {
			// defencing, find locker name
			ind := strings.LastIndex(output, lock_id) - 1
			for i := ind; i >= 0; i-- {
				if output[i] == '\n' {
					imageLocker = output[(i + 1):ind]
					break
				}
			}
			// remove a lock: rbd lock remove
			cmd, err = b.plugin.execCommand("rbd",
				append([]string{"lock", "remove", b.Image, lock_id, imageLocker, "--pool", b.Pool, "--id", b.Id, "-m", mon}, secret_opt...))
		}

		if err == nil {
			//locking operation is done
			glog.V(3).Infof("rbd lock (locking %v) finished successfully on rbd image %v/%v", lock, b.Pool, b.Image)
			break
		}
	}
	return false, err
}

func (locker *RBDLocker) Defencing(c rbdMounter, hostName string) error {
	// no need to fence readOnly
	if c.ReadOnly {
		return nil
	}

	_, err := locker.rbdLock(c, hostName, false, false)
	return err
}

func (locker *RBDLocker) Fencing(b rbdMounter, hostName string) error {
	// no need to fence readOnly
	if (&b).GetAttributes().ReadOnly {
		return nil
	}
	_, err := locker.rbdLock(b, hostName, true, false)
	return err
}

func (locker *RBDLocker) IsLocked(c rbdMounter, hostName string) (bool, error) {
	return locker.rbdLock(c, hostName, true, true)
}
