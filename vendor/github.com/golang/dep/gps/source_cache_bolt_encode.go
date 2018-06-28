// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"encoding/binary"
	"time"

	"github.com/boltdb/bolt"
	"github.com/golang/dep/gps/internal/pb"
	"github.com/golang/dep/gps/pkgtree"
	"github.com/golang/protobuf/proto"
	"github.com/jmank88/nuts"
	"github.com/pkg/errors"
)

var (
	cacheKeyComment    = []byte("c")
	cacheKeyConstraint = cacheKeyComment
	cacheKeyError      = []byte("e")
	cacheKeyHash       = []byte("h")
	cacheKeyIgnored    = []byte("i")
	cacheKeyImport     = cacheKeyIgnored
	cacheKeyLock       = []byte("l")
	cacheKeyName       = []byte("n")
	cacheKeyOverride   = []byte("o")
	cacheKeyPTree      = []byte("p")
	cacheKeyRequired   = []byte("r")
	cacheKeyRevision   = cacheKeyRequired
	cacheKeyTestImport = []byte("t")

	cacheRevision = byte('r')
	cacheVersion  = byte('v')
)

// propertiesFromCache returns a new ProjectRoot and ProjectProperties with the fields from m.
func propertiesFromCache(m *pb.ProjectProperties) (ProjectRoot, ProjectProperties, error) {
	ip := ProjectRoot(m.Root)
	var pp ProjectProperties
	pp.Source = m.Source

	if m.Constraint == nil {
		pp.Constraint = Any()
	} else {
		c, err := constraintFromCache(m.Constraint)
		if err != nil {
			return "", ProjectProperties{}, err
		}
		pp.Constraint = c
	}

	return ip, pp, nil
}

// projectPropertiesMsgs is a convenience tuple.
type projectPropertiesMsgs struct {
	pp pb.ProjectProperties
	c  pb.Constraint
}

// copyFrom sets the ProjectPropertiesMsg fields from ip and pp.
func (ms *projectPropertiesMsgs) copyFrom(ip ProjectRoot, pp ProjectProperties) {
	ms.pp.Root = string(ip)
	ms.pp.Source = pp.Source

	if pp.Constraint != nil && !IsAny(pp.Constraint) {
		pp.Constraint.copyTo(&ms.c)
		ms.pp.Constraint = &ms.c
	} else {
		ms.pp.Constraint = nil
	}
}

// cachePutManifest stores a Manifest in the bolt.Bucket.
func cachePutManifest(b *bolt.Bucket, m Manifest) error {
	var ppMsg projectPropertiesMsgs

	constraints := m.DependencyConstraints()
	if len(constraints) > 0 {
		cs, err := b.CreateBucket(cacheKeyConstraint)
		if err != nil {
			return err
		}
		key := make(nuts.Key, nuts.KeyLen(uint64(len(constraints)-1)))
		var i uint64
		for ip, pp := range constraints {
			ppMsg.copyFrom(ip, pp)
			v, err := proto.Marshal(&ppMsg.pp)
			if err != nil {
				return err
			}
			key.Put(i)
			i++
			if err := cs.Put(key, v); err != nil {
				return err
			}
		}
	}

	rm, ok := m.(RootManifest)
	if !ok {
		return nil
	}

	ignored := rm.IgnoredPackages().ToSlice()
	if len(ignored) > 0 {
		ig, err := b.CreateBucket(cacheKeyIgnored)
		if err != nil {
			return err
		}
		key := make(nuts.Key, nuts.KeyLen(uint64(len(ignored)-1)))
		var i uint64
		for _, ip := range ignored {
			key.Put(i)
			i++
			if err := ig.Put(key, []byte(ip)); err != nil {
				return err
			}
		}
	}

	overrides := rm.Overrides()
	if len(overrides) > 0 {
		ovr, err := b.CreateBucket(cacheKeyOverride)
		if err != nil {
			return err
		}
		key := make(nuts.Key, nuts.KeyLen(uint64(len(overrides)-1)))
		var i uint64
		for ip, pp := range overrides {
			ppMsg.copyFrom(ip, pp)
			v, err := proto.Marshal(&ppMsg.pp)
			if err != nil {
				return err
			}
			key.Put(i)
			i++
			if err := ovr.Put(key, v); err != nil {
				return err
			}
		}
	}

	required := rm.RequiredPackages()
	if len(required) > 0 {
		req, err := b.CreateBucket(cacheKeyRequired)
		if err != nil {
			return err
		}
		key := make(nuts.Key, nuts.KeyLen(uint64(len(required)-1)))
		var i uint64
		for ip, ok := range required {
			if ok {
				key.Put(i)
				i++
				if err := req.Put(key, []byte(ip)); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// cacheGetManifest returns a new RootManifest with the data retrieved from the bolt.Bucket.
func cacheGetManifest(b *bolt.Bucket) (RootManifest, error) {
	//TODO consider storing slice/map lens to enable calling make() with capacity
	m := &simpleRootManifest{
		c:   make(ProjectConstraints),
		ovr: make(ProjectConstraints),
		req: make(map[string]bool),
	}

	// Constraints
	if cs := b.Bucket(cacheKeyConstraint); cs != nil {
		var msg pb.ProjectProperties
		err := cs.ForEach(func(_, v []byte) error {
			if err := proto.Unmarshal(v, &msg); err != nil {
				return err
			}
			ip, pp, err := propertiesFromCache(&msg)
			if err != nil {
				return err
			}
			m.c[ip] = pp
			return nil
		})
		if err != nil {
			return nil, errors.Wrap(err, "failed to get constraints")
		}
	}

	// Ignored
	if ig := b.Bucket(cacheKeyIgnored); ig != nil {
		var igslice []string
		err := ig.ForEach(func(_, v []byte) error {
			igslice = append(igslice, string(v))
			return nil
		})
		m.ig = pkgtree.NewIgnoredRuleset(igslice)
		if err != nil {
			return nil, errors.Wrap(err, "failed to get ignored")
		}
	}

	// Overrides
	if os := b.Bucket(cacheKeyOverride); os != nil {
		var msg pb.ProjectProperties
		err := os.ForEach(func(_, v []byte) error {
			if err := proto.Unmarshal(v, &msg); err != nil {
				return err
			}
			ip, pp, err := propertiesFromCache(&msg)
			if err != nil {
				return err
			}
			m.ovr[ip] = pp
			return nil
		})
		if err != nil {
			return nil, errors.Wrap(err, "failed to get overrides")
		}
	}

	// Required
	if req := b.Bucket(cacheKeyRequired); req != nil {
		err := req.ForEach(func(_, v []byte) error {
			m.req[string(v)] = true
			return nil
		})
		if err != nil {
			return nil, errors.Wrap(err, "failed to get required")
		}
	}

	return m, nil
}

// copyTo returns a serializable representation of lp.
func (lp LockedProject) copyTo(msg *pb.LockedProject, c *pb.Constraint) {
	if lp.v == nil {
		msg.UnpairedVersion = nil
	} else {
		lp.v.copyTo(c)
		msg.UnpairedVersion = c
	}
	msg.Root = string(lp.pi.ProjectRoot)
	msg.Source = lp.pi.Source
	msg.Revision = string(lp.r)
	msg.Packages = lp.pkgs
}

// lockedProjectFromCache returns a new LockedProject with fields from m.
func lockedProjectFromCache(m *pb.LockedProject) (LockedProject, error) {
	var uv UnpairedVersion
	var err error
	if m.UnpairedVersion != nil {
		uv, err = unpairedVersionFromCache(m.UnpairedVersion)
		if err != nil {
			return LockedProject{}, err
		}
	}
	return LockedProject{
		pi: ProjectIdentifier{
			ProjectRoot: ProjectRoot(m.Root),
			Source:      m.Source,
		},
		v:    uv,
		r:    Revision(m.Revision),
		pkgs: m.Packages,
	}, nil
}

// cachePutLock stores the Lock as fields in the bolt.Bucket.
func cachePutLock(b *bolt.Bucket, l Lock) error {
	// InputHash
	if v := l.InputsDigest(); len(v) > 0 {
		if err := b.Put(cacheKeyHash, v); err != nil {
			return errors.Wrap(err, "failed to put hash")
		}
	}

	// Projects
	if projects := l.Projects(); len(projects) > 0 {
		lb, err := b.CreateBucket(cacheKeyLock)
		if err != nil {
			return err
		}
		key := make(nuts.Key, nuts.KeyLen(uint64(len(projects)-1)))
		var msg pb.LockedProject
		var cMsg pb.Constraint
		for i, lp := range projects {
			lp.copyTo(&msg, &cMsg)
			v, err := proto.Marshal(&msg)
			if err != nil {
				return err
			}
			key.Put(uint64(i))
			if err := lb.Put(key, v); err != nil {
				return err
			}
		}
	}

	return nil
}

// cacheGetLock returns a new *safeLock with the fields retrieved from the bolt.Bucket.
func cacheGetLock(b *bolt.Bucket) (*safeLock, error) {
	l := &safeLock{
		h: b.Get(cacheKeyHash),
	}
	if locked := b.Bucket(cacheKeyLock); locked != nil {
		var msg pb.LockedProject
		err := locked.ForEach(func(_, v []byte) error {
			if err := proto.Unmarshal(v, &msg); err != nil {
				return err
			}
			lp, err := lockedProjectFromCache(&msg)
			if err != nil {
				return err
			}
			l.p = append(l.p, lp)
			return nil
		})
		if err != nil {
			return nil, errors.Wrap(err, "failed to get locked projects")
		}
	}
	return l, nil
}

// cachePutPackageOrError stores the pkgtree.PackageOrErr as fields in the bolt.Bucket.
func cachePutPackageOrErr(b *bolt.Bucket, poe pkgtree.PackageOrErr) error {
	if poe.Err != nil {
		err := b.Put(cacheKeyError, []byte(poe.Err.Error()))
		return errors.Wrapf(err, "failed to put error: %v", poe.Err)
	}
	if len(poe.P.CommentPath) > 0 {
		err := b.Put(cacheKeyComment, []byte(poe.P.CommentPath))
		if err != nil {
			return errors.Wrapf(err, "failed to put package: %v", poe.P)
		}
	}
	if len(poe.P.Imports) > 0 {
		ip, err := b.CreateBucket(cacheKeyImport)
		if err != nil {
			return err
		}
		key := make(nuts.Key, nuts.KeyLen(uint64(len(poe.P.Imports)-1)))
		for i := range poe.P.Imports {
			v := []byte(poe.P.Imports[i])
			key.Put(uint64(i))
			if err := ip.Put(key, v); err != nil {
				return err
			}
		}
	}

	if len(poe.P.Name) > 0 {
		err := b.Put(cacheKeyName, []byte(poe.P.Name))
		if err != nil {
			return errors.Wrapf(err, "failed to put package: %v", poe.P)
		}
	}

	if len(poe.P.TestImports) > 0 {
		ip, err := b.CreateBucket(cacheKeyTestImport)
		if err != nil {
			return err
		}
		key := make(nuts.Key, nuts.KeyLen(uint64(len(poe.P.TestImports)-1)))
		for i := range poe.P.TestImports {
			v := []byte(poe.P.TestImports[i])
			key.Put(uint64(i))
			if err := ip.Put(key, v); err != nil {
				return err
			}
		}
	}
	return nil
}

// cacheGetPackageOrErr returns a new pkgtree.PackageOrErr with fields retrieved
// from the bolt.Bucket.
func cacheGetPackageOrErr(b *bolt.Bucket) (pkgtree.PackageOrErr, error) {
	if v := b.Get(cacheKeyError); len(v) > 0 {
		return pkgtree.PackageOrErr{
			Err: errors.New(string(v)),
		}, nil
	}

	var p pkgtree.Package
	p.CommentPath = string(b.Get(cacheKeyComment))
	if ip := b.Bucket(cacheKeyImport); ip != nil {
		err := ip.ForEach(func(_, v []byte) error {
			p.Imports = append(p.Imports, string(v))
			return nil
		})
		if err != nil {
			return pkgtree.PackageOrErr{}, err
		}
	}
	p.Name = string(b.Get(cacheKeyName))
	if tip := b.Bucket(cacheKeyTestImport); tip != nil {
		err := tip.ForEach(func(_, v []byte) error {
			p.TestImports = append(p.TestImports, string(v))
			return nil
		})
		if err != nil {
			return pkgtree.PackageOrErr{}, err
		}
	}
	return pkgtree.PackageOrErr{P: p}, nil
}

// cacheTimestampedKey returns a prefixed key with a trailing timestamp.
func cacheTimestampedKey(pre byte, t time.Time) []byte {
	b := make([]byte, 9)
	b[0] = pre
	binary.BigEndian.PutUint64(b[1:], uint64(t.Unix()))
	return b
}

// boltTxOrBucket is a minimal interface satisfied by bolt.Tx and bolt.Bucket.
type boltTxOrBucket interface {
	Cursor() *bolt.Cursor
	DeleteBucket([]byte) error
	Bucket([]byte) *bolt.Bucket
}

// cachePrefixDelete prefix scans and deletes each bucket.
func cachePrefixDelete(tob boltTxOrBucket, pre byte) error {
	c := tob.Cursor()
	for k, _ := c.Seek([]byte{pre}); len(k) > 0 && k[0] == pre; k, _ = c.Next() {
		if err := tob.DeleteBucket(k); err != nil {
			return errors.Wrapf(err, "failed to delete bucket: %s", k)
		}
	}
	return nil
}

// cacheFindLatestValid prefix scans for the latest bucket which is timestamped >= epoch,
// or returns nil if none exists.
func cacheFindLatestValid(tob boltTxOrBucket, pre byte, epoch int64) *bolt.Bucket {
	c := tob.Cursor()
	var latest []byte
	for k, _ := c.Seek([]byte{pre}); len(k) > 0 && k[0] == pre; k, _ = c.Next() {
		latest = k
	}
	if latest == nil {
		return nil
	}
	ts := latest[1:]
	if len(ts) != 8 {
		return nil
	}
	if int64(binary.BigEndian.Uint64(ts)) < epoch {
		return nil
	}
	return tob.Bucket(latest)
}
