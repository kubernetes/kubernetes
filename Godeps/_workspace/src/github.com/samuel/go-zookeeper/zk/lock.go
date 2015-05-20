package zk

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var (
	ErrDeadlock  = errors.New("zk: trying to acquire a lock twice")
	ErrNotLocked = errors.New("zk: not locked")
)

type Lock struct {
	c        *Conn
	path     string
	acl      []ACL
	lockPath string
	seq      int
}

func NewLock(c *Conn, path string, acl []ACL) *Lock {
	return &Lock{
		c:    c,
		path: path,
		acl:  acl,
	}
}

func parseSeq(path string) (int, error) {
	parts := strings.Split(path, "-")
	return strconv.Atoi(parts[len(parts)-1])
}

func (l *Lock) Lock() error {
	if l.lockPath != "" {
		return ErrDeadlock
	}

	prefix := fmt.Sprintf("%s/lock-", l.path)

	path := ""
	var err error
	for i := 0; i < 3; i++ {
		path, err = l.c.CreateProtectedEphemeralSequential(prefix, []byte{}, l.acl)
		if err == ErrNoNode {
			// Create parent node.
			parts := strings.Split(l.path, "/")
			pth := ""
			for _, p := range parts[1:] {
				pth += "/" + p
				_, err := l.c.Create(pth, []byte{}, 0, l.acl)
				if err != nil && err != ErrNodeExists {
					return err
				}
			}
		} else if err == nil {
			break
		} else {
			return err
		}
	}
	if err != nil {
		return err
	}

	seq, err := parseSeq(path)
	if err != nil {
		return err
	}

	for {
		children, _, err := l.c.Children(l.path)
		if err != nil {
			return err
		}

		lowestSeq := seq
		prevSeq := 0
		prevSeqPath := ""
		for _, p := range children {
			s, err := parseSeq(p)
			if err != nil {
				return err
			}
			if s < lowestSeq {
				lowestSeq = s
			}
			if s < seq && s > prevSeq {
				prevSeq = s
				prevSeqPath = p
			}
		}

		if seq == lowestSeq {
			// Acquired the lock
			break
		}

		// Wait on the node next in line for the lock
		_, _, ch, err := l.c.GetW(l.path + "/" + prevSeqPath)
		if err != nil && err != ErrNoNode {
			return err
		} else if err != nil && err == ErrNoNode {
			// try again
			continue
		}

		ev := <-ch
		if ev.Err != nil {
			return ev.Err
		}
	}

	l.seq = seq
	l.lockPath = path
	return nil
}

func (l *Lock) Unlock() error {
	if l.lockPath == "" {
		return ErrNotLocked
	}
	if err := l.c.Delete(l.lockPath, -1); err != nil {
		return err
	}
	l.lockPath = ""
	l.seq = 0
	return nil
}
