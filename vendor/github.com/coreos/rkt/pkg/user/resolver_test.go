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

package user_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	osuser "os/user"

	"github.com/coreos/rkt/pkg/user"
)

const (
	maxInt = int(^uint(0) >> 1)
	minInt = -maxInt - 1
)

func TestNumeric(t *testing.T) {
	for i, tt := range []struct {
		id string

		// expected
		err      bool
		uid, gid int
	}{
		{
			id:  "foo",
			err: true,
		},
		{
			id:  "",
			err: true,
		},
		{
			id:  "root",
			uid: 0, gid: 0,
		},
		{
			id:  "0",
			uid: 0, gid: 0,
		},
		{
			id:  "1",
			uid: 1, gid: 1,
		},
		{
			id:  strconv.Itoa(minInt),
			uid: minInt, gid: minInt,
		},
		{
			id:  strconv.Itoa(maxInt),
			uid: maxInt, gid: maxInt,
		},
		{
			id:  "9223372036854775808", // overflow (64bit) int by one
			err: true,
		},
		{
			id:  "-9223372036854775809", // overflow (64bit) int by one
			err: true,
		},
		{
			id:  "-1",
			uid: -1, gid: -1,
		},
	} {
		gen, err := user.NumericIDs(tt.id)
		if err == nil && tt.err {
			t.Errorf("test %d: expected err but got none", i)
		}

		if err != nil {
			continue
		}

		uid, gid, err := gen.IDs()
		if err != nil {
			panic(err) // must not happen
		}

		if uid != tt.uid {
			t.Errorf("test %d: expected uid %d but got %d", i, tt.uid, uid)
		}

		if gid != tt.gid {
			t.Errorf("test %d: expected gid %d but got %d", i, tt.gid, gid)
		}
	}
}

func TestFromEtc(t *testing.T) {
	root, err := ioutil.TempDir("", "rkt-TestFromEtc-")
	if err != nil {
		panic(err)
	}

	defer os.RemoveAll(root)

	if err := os.Mkdir(
		filepath.Join(root, "etc"),
		0700,
	); err != nil {
		panic(err)
	}

	if err := ioutil.WriteFile(
		filepath.Join(root, "etc/passwd"),
		[]byte(`u1:xxx:1000:100:::`),
		0600,
	); err != nil {
		panic(err)
	}

	if err := ioutil.WriteFile(
		filepath.Join(root, "etc/group"),
		[]byte(`g1:xxx:100:u1`),
		0600,
	); err != nil {
		panic(err)
	}

	for i, tt := range []struct {
		username, group string

		// expected
		err      bool
		uid, gid int
	}{
		{
			uid: -1,
			gid: -1,
			err: false,
		},
		{
			username: "unknown",

			uid: -1,
			gid: -1,
			err: true,
		},
		{
			group: "unknown",

			uid: -1,
			gid: -1,
			err: true,
		},
		{
			username: "u1",

			uid: 1000,
			gid: -1,
			err: false,
		},
		{
			username: "u1",
			group:    "unknown",

			uid: 1000,
			gid: -1,
			err: true,
		},
		{
			group: "g1",

			uid: -1,
			gid: 100,
			err: false,
		},
		{
			username: "unknown",
			group:    "g1",

			uid: -1,
			gid: -1,
			err: true,
		},
		{
			username: "u1",
			group:    "g1",

			uid: 1000,
			gid: 100,
			err: false,
		},
	} {
		gen, err := user.IDsFromEtc(root, tt.username, tt.group)
		if err != nil {
			panic(err)
		}

		uid, gid, err := gen.IDs()
		if err == nil && tt.err {
			t.Errorf("test %d: expected err but got none", i)
		}

		if err != nil && !tt.err {
			t.Errorf("test %d: expected no err but got one", i)
		}

		if uid != tt.uid {
			t.Errorf("test %d: expected uid %d but got %d", i, tt.uid, uid)
		}

		if gid != tt.gid {
			t.Errorf("test %d: expected gid %d but got %d", i, tt.gid, gid)
		}
	}
}

func TestStat(t *testing.T) {
	tmp, err := ioutil.TempFile("", "rkt-TestStat-")
	if err != nil {
		panic(err)
	}

	defer os.Remove(tmp.Name())

	rng := user.NewBlankUidRange()
	rng.SetRandomUidRange(100)

	u, err := osuser.Current()
	if err != nil {
		panic(err)
	}

	procUid, err := strconv.Atoi(u.Uid)
	if err != nil {
		panic(err)
	}

	procGid, err := strconv.Atoi(u.Gid)
	if err != nil {
		panic(err)
	}

	for i, tt := range []struct {
		root, path string

		// expected
		errIDs, err bool
		uid, gid    int
	}{
		{
			root: "",
			path: "",

			err: true,
		},
		{
			root: "unknown",
			path: "",

			err: true,
		},
		{
			root: "",
			path: "unknown",

			err: true,
		},
		{
			root: "",
			path: tmp.Name(),

			uid: procUid,
			gid: procGid,
		},
		{
			root: "/",
			path: tmp.Name(),

			uid: procUid,
			gid: procGid,
		},
		{
			root: "unknown",
			path: tmp.Name(),

			errIDs: true,
			uid:    -1,
			gid:    -1,
		},
		{
			root: filepath.Dir(tmp.Name()),
			path: "",

			err: true,
		},
		{
			root: filepath.Dir(tmp.Name()),
			path: "/" + filepath.Base(tmp.Name()),

			uid: procUid,
			gid: procGid,
		},
		{
			root: filepath.Dir(tmp.Name()),
			path: "/unknown",

			errIDs: true,
			uid:    -1,
			gid:    -1,
		},
		{
			root: filepath.Dir(tmp.Name()),
			path: "unknown",

			err: true,
		},
	} {
		gen, err := user.IDsFromStat(tt.root, tt.path, nil)
		if err == nil && tt.err {
			t.Errorf("test %d: expected error but got one", i)
		}

		if err != nil {
			continue
		}

		uid, gid, err := gen.IDs()
		if err == nil && tt.errIDs {
			t.Errorf("test %d: expected err but got none", i)
		}

		if uid != tt.uid {
			t.Errorf("test %d: expected uid %d but got %d", i, tt.uid, uid)
		}

		if gid != tt.gid {
			t.Errorf("test %d: expected gid %d but got %d", i, tt.gid, gid)
		}
	}
}
