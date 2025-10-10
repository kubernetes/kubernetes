//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package users

import (
	"os"
	"reflect"
	"testing"
)

func TestParseLoginDef(t *testing.T) {
	testCases := []struct {
		name           string
		input          string
		expectedLimits *limits
		expectedError  bool
	}{
		{
			name:          "non number value for tracked limit",
			input:         "SYS_UID_MIN foo\n",
			expectedError: true,
		},
		{
			name:           "empty string must return defaults",
			expectedLimits: defaultLimits,
		},
		{
			name:           "no tracked limits in file must return defaults",
			input:          "# some comment\n",
			expectedLimits: defaultLimits,
		},
		{
			name:           "must parse all valid tracked limits",
			input:          "SYS_UID_MIN 101\nSYS_UID_MAX 998\nSYS_GID_MIN 102\nSYS_GID_MAX 999\n",
			expectedLimits: &limits{minUID: 101, maxUID: 998, minGID: 102, maxGID: 999},
		},
		{
			name:           "must return defaults for missing limits",
			input:          "SYS_UID_MIN 101\n#SYS_UID_MAX 998\nSYS_GID_MIN 102\n#SYS_GID_MAX 999\n",
			expectedLimits: &limits{minUID: 101, maxUID: defaultLimits.maxUID, minGID: 102, maxGID: defaultLimits.maxGID},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseLoginDefs(tc.input)
			if err != nil != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if err == nil && *tc.expectedLimits != *got {
				t.Fatalf("expected limits %+v, got %+v", tc.expectedLimits, got)
			}
		})
	}
}

func TestParseEntries(t *testing.T) {
	testCases := []struct {
		name            string
		file            string
		expectedEntries []*entry
		totalFields     int
		expectedError   bool
	}{
		{
			name:          "totalFields must be a known value",
			expectedError: true,
		},
		{
			name:          "unexpected number of fields",
			file:          "foo:x:100::::::",
			totalFields:   totalFieldsUser,
			expectedError: true,
		},
		{
			name:          "cannot parse 'bar' as UID",
			file:          "foo:x:bar:101:::\n",
			totalFields:   totalFieldsUser,
			expectedError: true,
		},
		{
			name:          "cannot parse 'bar' as GID",
			file:          "foo:x:101:bar:::\n",
			totalFields:   totalFieldsUser,
			expectedError: true,
		},
		{
			name:        "valid file for users",
			file:        "\nfoo:x:100:101:foo:/home/foo:/bin/bash\n\nbar:x:102:103:bar::\n",
			totalFields: totalFieldsUser,
			expectedEntries: []*entry{
				{name: "foo", id: 100, gid: 101, shell: "/bin/bash"},
				{name: "bar", id: 102, gid: 103},
			},
		},
		{
			name:        "valid file for groups",
			file:        "\nfoo:x:100:bar,baz\n\nbar:x:101:baz\n",
			totalFields: totalFieldsGroup,
			expectedEntries: []*entry{
				{name: "foo", id: 100, userNames: []string{"bar", "baz"}},
				{name: "bar", id: 101, userNames: []string{"baz"}},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseEntries(tc.file, tc.totalFields)
			if err != nil != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if err != nil {
				return
			}
			if len(tc.expectedEntries) != len(got) {
				t.Fatalf("expected entries %d, got %d", len(tc.expectedEntries), len(got))
			}
			for i := range got {
				if !reflect.DeepEqual(tc.expectedEntries[i], got[i]) {
					t.Fatalf("expected entry at position %d: %+v, got: %+v", i, tc.expectedEntries[i], got[i])
				}
			}
		})
	}
}

func TestValidateEntries(t *testing.T) {
	testCases := []struct {
		name           string
		users          []*entry
		groups         []*entry
		expectedUsers  []*entry
		expectedGroups []*entry
		expectedError  bool
	}{
		{
			name: "UID for user is outside of system limits",
			users: []*entry{
				{name: "kubeadm-etcd", id: 2000, gid: 102, shell: noshell},
			},
			groups:        []*entry{},
			expectedError: true,
		},
		{
			name: "user has unexpected shell",
			users: []*entry{
				{name: "kubeadm-etcd", id: 102, gid: 102, shell: "foo"},
			},
			groups:        []*entry{},
			expectedError: true,
		},
		{
			name: "user is mapped to unknown group",
			users: []*entry{
				{name: "kubeadm-etcd", id: 102, gid: 102, shell: noshell},
			},
			groups:        []*entry{},
			expectedError: true,
		},
		{
			name: "user and group names do not match",
			users: []*entry{
				{name: "kubeadm-etcd", id: 102, gid: 102, shell: noshell},
			},
			groups: []*entry{
				{name: "foo", id: 102},
			},
			expectedError: true,
		},
		{
			name:  "GID is outside system limits",
			users: []*entry{},
			groups: []*entry{
				{name: "kubeadm-etcd", id: 2000},
			},
			expectedError: true,
		},
		{
			name:  "group is missing users",
			users: []*entry{},
			groups: []*entry{
				{name: "kubeadm-etcd", id: 100},
			},
			expectedError: true,
		},
		{
			name:           "empty input must return default users and groups",
			users:          []*entry{},
			groups:         []*entry{},
			expectedUsers:  usersToCreateSpec,
			expectedGroups: groupsToCreateSpec,
		},
		{
			name: "existing valid users mapped to groups",
			users: []*entry{
				{name: "kubeadm-etcd", id: 100, gid: 102, shell: noshell},
				{name: "kubeadm-kas", id: 101, gid: 103, shell: noshell},
			},
			groups: []*entry{
				{name: "kubeadm-etcd", id: 102, userNames: []string{"kubeadm-etcd"}},
				{name: "kubeadm-kas", id: 103, userNames: []string{"kubeadm-kas"}},
				{name: "kubeadm-sa-key-readers", id: 104, userNames: []string{"kubeadm-kas", "kubeadm-kcm"}},
			},
			expectedUsers: []*entry{
				{name: "kubeadm-kcm"},
				{name: "kubeadm-ks"},
			},
			expectedGroups: []*entry{
				{name: "kubeadm-kcm", userNames: []string{"kubeadm-kcm"}},
				{name: "kubeadm-ks", userNames: []string{"kubeadm-ks"}},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			users, groups, err := validateEntries(tc.users, tc.groups, defaultLimits)
			if err != nil != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if err != nil {
				return
			}
			if len(tc.expectedUsers) != len(users) {
				t.Fatalf("expected users %d, got %d", len(tc.expectedUsers), len(users))
			}
			for i := range users {
				if !reflect.DeepEqual(tc.expectedUsers[i], users[i]) {
					t.Fatalf("expected user at position %d: %+v, got: %+v", i, tc.expectedUsers[i], users[i])
				}
			}
			if len(tc.expectedGroups) != len(groups) {
				t.Fatalf("expected groups %d, got %d", len(tc.expectedGroups), len(groups))
			}
			for i := range groups {
				if !reflect.DeepEqual(tc.expectedGroups[i], groups[i]) {
					t.Fatalf("expected group at position %d: %+v, got: %+v", i, tc.expectedGroups[i], groups[i])
				}
			}
		})
	}
}

func TestAllocateIDs(t *testing.T) {
	testCases := []struct {
		name          string
		entries       []*entry
		min           int64
		max           int64
		total         int
		expectedIDs   []int64
		expectedError bool
	}{
		{
			name:        "zero total ids returns empty slice",
			expectedIDs: []int64{},
		},
		{
			name: "not enough free ids in range",
			entries: []*entry{
				{name: "foo", id: 101},
				{name: "bar", id: 103},
				{name: "baz", id: 105},
			},
			min:           100,
			max:           105,
			total:         4,
			expectedError: true,
		},
		{
			name: "successfully allocate ids",
			entries: []*entry{
				{name: "foo", id: 101},
				{name: "bar", id: 103},
				{name: "baz", id: 105},
			},
			min:           100,
			max:           110,
			total:         4,
			expectedIDs:   []int64{100, 102, 104, 106},
			expectedError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := allocateIDs(tc.entries, tc.min, tc.max, tc.total)
			if err != nil != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if err != nil {
				return
			}
			if len(tc.expectedIDs) != len(got) {
				t.Fatalf("expected id %d, got %d", len(tc.expectedIDs), len(got))
			}
			for i := range got {
				if !reflect.DeepEqual(tc.expectedIDs[i], got[i]) {
					t.Fatalf("expected id at position %d: %+v, got: %+v", i, tc.expectedIDs[i], got[i])
				}
			}
		})
	}
}

func TestAddEntries(t *testing.T) {
	testCases := []struct {
		name           string
		file           string
		entries        []*entry
		createEntry    func(*entry) string
		expectedOutput string
	}{
		{
			name: "user entries are added",
			file: "foo:x:101:101:::/bin/false\n",
			entries: []*entry{
				{name: "bar", id: 102, gid: 102},
				{name: "baz", id: 103, gid: 103},
			},
			expectedOutput: "foo:x:101:101:::/bin/false\nbar:x:102:102:::/bin/false\nbaz:x:103:103:::/bin/false\n",
			createEntry:    createUser,
		},
		{
			name: "user entries are added (new line is appended)",
			file: "foo:x:101:101:::/bin/false",
			entries: []*entry{
				{name: "bar", id: 102, gid: 102},
			},
			expectedOutput: "foo:x:101:101:::/bin/false\nbar:x:102:102:::/bin/false\n",
			createEntry:    createUser,
		},
		{
			name: "group entries are added",
			file: "foo:x:101:foo\n",
			entries: []*entry{
				{name: "bar", id: 102, userNames: []string{"bar"}},
				{name: "baz", id: 103, userNames: []string{"baz"}},
			},
			expectedOutput: "foo:x:101:foo\nbar:x:102:bar\nbaz:x:103:baz\n",
			createEntry:    createGroup,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := addEntries(tc.file, tc.entries, tc.createEntry)
			if tc.expectedOutput != got {
				t.Fatalf("expected output:\n%s\ngot:\n%s\n", tc.expectedOutput, got)
			}
		})
	}
}

func TestRemoveEntries(t *testing.T) {
	testCases := []struct {
		name            string
		file            string
		entries         []*entry
		expectedRemoved int
		expectedOutput  string
	}{
		{
			name:            "entries that are missing do not cause an error",
			file:            "foo:x:102:102:::/bin/false\nbar:x:103:103:::/bin/false\n",
			entries:         []*entry{},
			expectedRemoved: 0,
			expectedOutput:  "foo:x:102:102:::/bin/false\nbar:x:103:103:::/bin/false\n",
		},
		{
			name: "user entry is removed",
			file: "foo:x:102:102:::/bin/false\nbar:x:103:103:::/bin/false\n",
			entries: []*entry{
				{name: "bar"},
			},
			expectedRemoved: 1,
			expectedOutput:  "foo:x:102:102:::/bin/false\n",
		},
		{
			name: "group entry is removed",
			file: "foo:x:102:foo\nbar:x:102:bar\n",
			entries: []*entry{
				{name: "bar"},
			},
			expectedRemoved: 1,
			expectedOutput:  "foo:x:102:foo\n",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, removed := removeEntries(tc.file, tc.entries)
			if tc.expectedRemoved != removed {
				t.Fatalf("expected entries to be removed: %v, got: %v", tc.expectedRemoved, removed)
			}
			if tc.expectedOutput != got {
				t.Fatalf("expected output:\n%s\ngot:\n%s\n", tc.expectedOutput, got)
			}
		})
	}
}

func TestAssignUserAndGroupIDs(t *testing.T) {
	testCases := []struct {
		name           string
		users          []*entry
		groups         []*entry
		usersToCreate  []*entry
		groupsToCreate []*entry
		uids           []int64
		gids           []int64
		expectedUsers  []*entry
		expectedGroups []*entry
		expectedError  bool
	}{
		{
			name: "not enough UIDs",
			usersToCreate: []*entry{
				{name: "foo"},
				{name: "bar"},
			},
			uids:          []int64{100},
			expectedError: true,
		},
		{
			name: "not enough GIDs",
			groupsToCreate: []*entry{
				{name: "foo"},
				{name: "bar"},
			},
			gids:          []int64{100},
			expectedError: true,
		},
		{
			name: "valid UIDs and GIDs are assigned to input",
			groups: []*entry{
				{name: "foo", id: 110},
				{name: "bar", id: 111},
			},
			usersToCreate: []*entry{
				{name: "foo"},
				{name: "bar"},
				{name: "baz"},
			},
			groupsToCreate: []*entry{
				{name: "baz"},
			},
			uids: []int64{100, 101, 102},
			gids: []int64{112},
			expectedUsers: []*entry{
				{name: "foo", id: 100, gid: 110},
				{name: "bar", id: 101, gid: 111},
				{name: "baz", id: 102, gid: 112},
			},
			expectedGroups: []*entry{
				{name: "baz", id: 112},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := assignUserAndGroupIDs(tc.groups, tc.usersToCreate, tc.groupsToCreate, tc.uids, tc.gids)
			if err != nil != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if err != nil {
				return
			}
			if len(tc.expectedUsers) != len(tc.usersToCreate) {
				t.Fatalf("expected users %d, got %d", len(tc.expectedUsers), len(tc.usersToCreate))
			}
			for i := range tc.usersToCreate {
				if !reflect.DeepEqual(tc.expectedUsers[i], tc.usersToCreate[i]) {
					t.Fatalf("expected user at position %d: %+v, got: %+v", i, tc.expectedUsers[i], tc.usersToCreate[i])
				}
			}
			if len(tc.expectedGroups) != len(tc.groupsToCreate) {
				t.Fatalf("expected groups %d, got %d", len(tc.expectedGroups), len(tc.groupsToCreate))
			}
			for i := range tc.groupsToCreate {
				if !reflect.DeepEqual(tc.expectedGroups[i], tc.groupsToCreate[i]) {
					t.Fatalf("expected group at position %d: %+v, got: %+v", i, tc.expectedGroups[i], tc.groupsToCreate[i])
				}
			}
		})
	}
}

func TestID(t *testing.T) {
	e := &entry{name: "foo", id: 101}
	m := &EntryMap{entries: map[string]*entry{
		"foo": e,
	}}
	id := m.ID("foo")
	if *id != 101 {
		t.Fatalf("expected: id=%d; got: id=%d", 101, *id)
	}
	id = m.ID("bar")
	if id != nil {
		t.Fatalf("expected nil for unknown entry")
	}
}

func TestAddUsersAndGroupsImpl(t *testing.T) {
	const (
		loginDef       = "SYS_UID_MIN 101\nSYS_UID_MAX 998\nSYS_GID_MIN 102\nSYS_GID_MAX 999\n"
		passwd         = "root:x:0:0:::/bin/bash\nkubeadm-etcd:x:101:102:::/bin/false\n"
		group          = "root:x:0:root\nkubeadm-etcd:x:102:kubeadm-etcd\n"
		expectedUsers  = "kubeadm-etcd{101,102};kubeadm-kas{102,103};kubeadm-kcm{103,104};kubeadm-ks{104,105};"
		expectedGroups = "kubeadm-etcd{102,0};kubeadm-kas{103,0};kubeadm-kcm{104,0};kubeadm-ks{105,0};kubeadm-sa-key-readers{106,0};"
	)
	fileLoginDef, close := writeTempFile(t, loginDef)
	defer close()
	filePasswd, close := writeTempFile(t, passwd)
	defer close()
	fileGroup, close := writeTempFile(t, group)
	defer close()
	got, err := addUsersAndGroupsImpl(fileLoginDef, filePasswd, fileGroup)
	if err != nil {
		t.Fatalf("AddUsersAndGroups failed: %v", err)
	}
	if expectedUsers != got.Users.String() {
		t.Fatalf("expected users: %q, got: %q", expectedUsers, got.Users.String())
	}
	if expectedGroups != got.Groups.String() {
		t.Fatalf("expected groups: %q, got: %q", expectedGroups, got.Groups.String())
	}
}

func TestRemoveUsersAndGroups(t *testing.T) {
	const (
		passwd         = "root:x:0:0:::/bin/bash\nkubeadm-etcd:x:101:102:::/bin/false\n"
		group          = "root:x:0:root\nkubeadm-etcd:x:102:kubeadm-etcd\n"
		expectedPasswd = "root:x:0:0:::/bin/bash\n"
		expectedGroup  = "root:x:0:root\n"
	)
	filePasswd, close := writeTempFile(t, passwd)
	defer close()
	fileGroup, close := writeTempFile(t, group)
	defer close()
	if err := removeUsersAndGroupsImpl(filePasswd, fileGroup); err != nil {
		t.Fatalf("RemoveUsersAndGroups failed: %v", err)
	}
	contentsPasswd := readTempFile(t, filePasswd)
	if expectedPasswd != contentsPasswd {
		t.Fatalf("expected passwd:\n%s\ngot:\n%s\n", expectedPasswd, contentsPasswd)
	}
	contentsGroup := readTempFile(t, fileGroup)
	if expectedGroup != contentsGroup {
		t.Fatalf("expected passwd:\n%s\ngot:\n%s\n", expectedGroup, contentsGroup)
	}
}

func writeTempFile(t *testing.T, contents string) (string, func()) {
	file, err := os.CreateTemp("", "")
	if err != nil {
		t.Fatalf("could not create file: %v", err)
	}
	if err := os.WriteFile(file.Name(), []byte(contents), os.ModePerm); err != nil {
		t.Fatalf("could not write file: %v", err)
	}
	close := func() {
		os.Remove(file.Name())
	}
	return file.Name(), close
}

func readTempFile(t *testing.T, path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("could not read file: %v", err)
	}
	return string(b)
}
