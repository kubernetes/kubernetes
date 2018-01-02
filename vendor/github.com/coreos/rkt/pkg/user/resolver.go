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

package user

import (
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/coreos/rkt/pkg/group"
	"github.com/coreos/rkt/pkg/passwd"
	"github.com/hashicorp/errwrap"
)

// Resolver defines the interface for resolving a UID/GID.
type Resolver interface {
	IDs() (uid int, gid int, err error)
}

type idsFromEtc struct {
	rootPath string
	username string
	group    string
}

// IDsFromEtc returns a new UID/GID resolver by parsing etc/passwd, and etc/group
// relative from the given rootPath looking for the given username, or group.
// If username is empty string the etc/passwd lookup will be omitted.
// If group is empty string the etc/group lookup will be omitted.
func IDsFromEtc(rootPath, username, group string) (Resolver, error) {
	return idsFromEtc{
		rootPath: rootPath,
		username: username,
		group:    group,
	}, nil
}

func (e idsFromEtc) IDs() (uid int, gid int, err error) {
	uid, gid = -1, -1

	uid, err = passwd.LookupUidFromFile(
		e.username,
		filepath.Join(e.rootPath, "etc/passwd"),
	)

	if e.username != "" && err != nil {
		return
	}

	gid, err = group.LookupGidFromFile(
		e.group,
		filepath.Join(e.rootPath, "etc/group"),
	)

	if e.group != "" && err != nil {
		return
	}

	return uid, gid, nil
}

type idsFromStat struct {
	path string
	r    *UidRange
}

// IDsFromStat returns a new UID/GID resolver deriving the UID/GID from file attributes
// and unshifts the UID/GID if the given range is not nil.
// If the given id does not start with a slash "/" an error is returned.
func IDsFromStat(rootPath, file string, r *UidRange) (Resolver, error) {
	if strings.HasPrefix(file, "/") {
		return idsFromStat{filepath.Join(rootPath, file), r}, nil
	}

	return nil, fmt.Errorf("invalid filename %q", file)
}

func (s idsFromStat) IDs() (int, int, error) {
	var stat syscall.Stat_t

	if err := syscall.Lstat(s.path, &stat); err != nil {
		return -1, -1, errwrap.Wrap(
			fmt.Errorf("unable to stat file %q", s.path),
			err,
		)
	}

	if s.r == nil {
		return int(stat.Uid), int(stat.Gid), nil
	}

	uid, _, err := s.r.UnshiftRange(stat.Uid, stat.Gid)
	if err != nil {
		return -1, -1, errwrap.Wrap(errors.New("unable to determine real uid"), err)
	}

	_, gid, err := s.r.UnshiftRange(stat.Uid, stat.Gid)
	if err != nil {
		return -1, -1, errwrap.Wrap(errors.New("unable to determine real gid"), err)
	}

	return int(uid), int(gid), nil
}

// numericIDs is the struct that always resolves to uid=i and gid=i.
type numericIDs struct {
	i int
}

// NumericIDs returns a resolver that will resolve constant UID/GID values.
// If the given id equals to "root" the resolver always resolves UID=0 and GID=0.
// If the given id is a numeric literal i it always resolves UID=i and GID=i.
// If the given id is neither "root" nor a numeric literal an error is returned.
func NumericIDs(id string) (Resolver, error) {
	if id == "root" {
		return numericIDs{0}, nil
	}

	if i, err := strconv.Atoi(id); err == nil {
		return numericIDs{i}, nil
	}

	return nil, fmt.Errorf("invalid id %q", id)
}

func (n numericIDs) IDs() (int, int, error) {
	return n.i, n.i, nil
}
