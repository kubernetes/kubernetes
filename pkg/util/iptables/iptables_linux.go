// +build linux

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

package iptables

import (
	"fmt"
	"net"
	"os"
	"time"

	"golang.org/x/sys/unix"
	"k8s.io/apimachinery/pkg/util/wait"
)

type locker struct {
	lock16 *os.File
	lock14 *net.UnixListener
}

func (l *locker) Close() {
	if l.lock16 != nil {
		l.lock16.Close()
	}
	if l.lock14 != nil {
		l.lock14.Close()
	}
}

func grabIptablesLocks(lockfilePath string) (iptablesLocker, error) {
	var err error
	var success bool

	l := &locker{}
	defer func(l *locker) {
		// Clean up immediately on failure
		if !success {
			l.Close()
		}
	}(l)

	// Grab both 1.6.x and 1.4.x-style locks; we don't know what the
	// iptables-restore version is if it doesn't support --wait, so we
	// can't assume which lock method it'll use.

	// Roughly duplicate iptables 1.6.x xtables_lock() function.
	l.lock16, err = os.OpenFile(lockfilePath, os.O_CREATE, 0600)
	if err != nil {
		return nil, fmt.Errorf("failed to open iptables lock %s: %v", lockfilePath, err)
	}

	if err := wait.PollImmediate(200*time.Millisecond, 2*time.Second, func() (bool, error) {
		if err := grabIptablesFileLock(l.lock16); err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, fmt.Errorf("failed to acquire new iptables lock: %v", err)
	}

	// Roughly duplicate iptables 1.4.x xtables_lock() function.
	if err := wait.PollImmediate(200*time.Millisecond, 2*time.Second, func() (bool, error) {
		l.lock14, err = net.ListenUnix("unix", &net.UnixAddr{Name: "@xtables", Net: "unix"})
		if err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, fmt.Errorf("failed to acquire old iptables lock: %v", err)
	}

	success = true
	return l, nil
}

func grabIptablesFileLock(f *os.File) error {
	return unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
}
