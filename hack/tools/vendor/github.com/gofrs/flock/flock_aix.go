// Copyright 2019 Tim Heckman. All rights reserved. Use of this source code is
// governed by the BSD 3-Clause license that can be found in the LICENSE file.

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code implements the filelock API using POSIX 'fcntl' locks, which attach
// to an (inode, process) pair rather than a file descriptor. To avoid unlocking
// files prematurely when the same file is opened through different descriptors,
// we allow only one read-lock at a time.
//
// This code is adapted from the Go package:
// cmd/go/internal/lockedfile/internal/filelock

//+build aix

package flock

import (
	"errors"
	"io"
	"os"
	"sync"
	"syscall"

	"golang.org/x/sys/unix"
)

type lockType int16

const (
	readLock  lockType = unix.F_RDLCK
	writeLock lockType = unix.F_WRLCK
)

type cmdType int

const (
	tryLock  cmdType = unix.F_SETLK
	waitLock cmdType = unix.F_SETLKW
)

type inode = uint64

type inodeLock struct {
	owner *Flock
	queue []<-chan *Flock
}

var (
	mu     sync.Mutex
	inodes = map[*Flock]inode{}
	locks  = map[inode]inodeLock{}
)

// Lock is a blocking call to try and take an exclusive file lock. It will wait
// until it is able to obtain the exclusive file lock. It's recommended that
// TryLock() be used over this function. This function may block the ability to
// query the current Locked() or RLocked() status due to a RW-mutex lock.
//
// If we are already exclusive-locked, this function short-circuits and returns
// immediately assuming it can take the mutex lock.
//
// If the *Flock has a shared lock (RLock), this may transparently replace the
// shared lock with an exclusive lock on some UNIX-like operating systems. Be
// careful when using exclusive locks in conjunction with shared locks
// (RLock()), because calling Unlock() may accidentally release the exclusive
// lock that was once a shared lock.
func (f *Flock) Lock() error {
	return f.lock(&f.l, writeLock)
}

// RLock is a blocking call to try and take a shared file lock. It will wait
// until it is able to obtain the shared file lock. It's recommended that
// TryRLock() be used over this function. This function may block the ability to
// query the current Locked() or RLocked() status due to a RW-mutex lock.
//
// If we are already shared-locked, this function short-circuits and returns
// immediately assuming it can take the mutex lock.
func (f *Flock) RLock() error {
	return f.lock(&f.r, readLock)
}

func (f *Flock) lock(locked *bool, flag lockType) error {
	f.m.Lock()
	defer f.m.Unlock()

	if *locked {
		return nil
	}

	if f.fh == nil {
		if err := f.setFh(); err != nil {
			return err
		}
		defer f.ensureFhState()
	}

	if _, err := f.doLock(waitLock, flag, true); err != nil {
		return err
	}

	*locked = true
	return nil
}

func (f *Flock) doLock(cmd cmdType, lt lockType, blocking bool) (bool, error) {
	// POSIX locks apply per inode and process, and the lock for an inode is
	// released when *any* descriptor for that inode is closed. So we need to
	// synchronize access to each inode internally, and must serialize lock and
	// unlock calls that refer to the same inode through different descriptors.
	fi, err := f.fh.Stat()
	if err != nil {
		return false, err
	}
	ino := inode(fi.Sys().(*syscall.Stat_t).Ino)

	mu.Lock()
	if i, dup := inodes[f]; dup && i != ino {
		mu.Unlock()
		return false, &os.PathError{
			Path: f.Path(),
			Err:  errors.New("inode for file changed since last Lock or RLock"),
		}
	}

	inodes[f] = ino

	var wait chan *Flock
	l := locks[ino]
	if l.owner == f {
		// This file already owns the lock, but the call may change its lock type.
	} else if l.owner == nil {
		// No owner: it's ours now.
		l.owner = f
	} else if !blocking {
		// Already owned: cannot take the lock.
		mu.Unlock()
		return false, nil
	} else {
		// Already owned: add a channel to wait on.
		wait = make(chan *Flock)
		l.queue = append(l.queue, wait)
	}
	locks[ino] = l
	mu.Unlock()

	if wait != nil {
		wait <- f
	}

	err = setlkw(f.fh.Fd(), cmd, lt)

	if err != nil {
		f.doUnlock()
		if cmd == tryLock && err == unix.EACCES {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

func (f *Flock) Unlock() error {
	f.m.Lock()
	defer f.m.Unlock()

	// if we aren't locked or if the lockfile instance is nil
	// just return a nil error because we are unlocked
	if (!f.l && !f.r) || f.fh == nil {
		return nil
	}

	if err := f.doUnlock(); err != nil {
		return err
	}

	f.fh.Close()

	f.l = false
	f.r = false
	f.fh = nil

	return nil
}

func (f *Flock) doUnlock() (err error) {
	var owner *Flock
	mu.Lock()
	ino, ok := inodes[f]
	if ok {
		owner = locks[ino].owner
	}
	mu.Unlock()

	if owner == f {
		err = setlkw(f.fh.Fd(), waitLock, unix.F_UNLCK)
	}

	mu.Lock()
	l := locks[ino]
	if len(l.queue) == 0 {
		// No waiters: remove the map entry.
		delete(locks, ino)
	} else {
		// The first waiter is sending us their file now.
		// Receive it and update the queue.
		l.owner = <-l.queue[0]
		l.queue = l.queue[1:]
		locks[ino] = l
	}
	delete(inodes, f)
	mu.Unlock()

	return err
}

// TryLock is the preferred function for taking an exclusive file lock. This
// function takes an RW-mutex lock before it tries to lock the file, so there is
// the possibility that this function may block for a short time if another
// goroutine is trying to take any action.
//
// The actual file lock is non-blocking. If we are unable to get the exclusive
// file lock, the function will return false instead of waiting for the lock. If
// we get the lock, we also set the *Flock instance as being exclusive-locked.
func (f *Flock) TryLock() (bool, error) {
	return f.try(&f.l, writeLock)
}

// TryRLock is the preferred function for taking a shared file lock. This
// function takes an RW-mutex lock before it tries to lock the file, so there is
// the possibility that this function may block for a short time if another
// goroutine is trying to take any action.
//
// The actual file lock is non-blocking. If we are unable to get the shared file
// lock, the function will return false instead of waiting for the lock. If we
// get the lock, we also set the *Flock instance as being share-locked.
func (f *Flock) TryRLock() (bool, error) {
	return f.try(&f.r, readLock)
}

func (f *Flock) try(locked *bool, flag lockType) (bool, error) {
	f.m.Lock()
	defer f.m.Unlock()

	if *locked {
		return true, nil
	}

	if f.fh == nil {
		if err := f.setFh(); err != nil {
			return false, err
		}
		defer f.ensureFhState()
	}

	haslock, err := f.doLock(tryLock, flag, false)
	if err != nil {
		return false, err
	}

	*locked = haslock
	return haslock, nil
}

// setlkw calls FcntlFlock with cmd for the entire file indicated by fd.
func setlkw(fd uintptr, cmd cmdType, lt lockType) error {
	for {
		err := unix.FcntlFlock(fd, int(cmd), &unix.Flock_t{
			Type:   int16(lt),
			Whence: io.SeekStart,
			Start:  0,
			Len:    0, // All bytes.
		})
		if err != unix.EINTR {
			return err
		}
	}
}
