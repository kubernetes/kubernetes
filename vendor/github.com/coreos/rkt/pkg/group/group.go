// Copyright 2015 The rkt Authors
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

package group

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/hashicorp/errwrap"
)

const (
	groupFilePath = "/etc/group"
)

// Group represents an entry in the group file.
type Group struct {
	Name  string
	Pass  string
	Gid   int
	Users []string
}

// LookupGid reads the group file specified by groupFile, and returns the gid of the group
// specified by groupName.
func LookupGidFromFile(groupName, groupFile string) (gid int, err error) {
	groups, err := parseGroupFile(groupFile)
	if err != nil {
		return -1, errwrap.Wrap(fmt.Errorf("error parsing %q file", groupFile), err)
	}

	group, ok := groups[groupName]
	if !ok {
		return -1, fmt.Errorf("%q group not found", groupName)
	}

	return group.Gid, nil
}

// LookupGid reads the group file and returns the gid of the group
// specified by groupName.
func LookupGid(groupName string) (gid int, err error) {
	return LookupGidFromFile(groupName, groupFilePath)
}

func parseGroupFile(path string) (group map[string]*Group, err error) {
	groupFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer groupFile.Close()

	return parseGroups(groupFile)
}

func parseGroups(r io.Reader) (group map[string]*Group, err error) {
	s := bufio.NewScanner(r)
	out := make(map[string]*Group)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}

		text := s.Text()
		if text == "" {
			continue
		}

		p, err := parseGroupLine(text)
		if err != nil {
			return nil, errwrap.Wrap(errors.New("error parsing line"), err)
		}

		out[p.Name] = p
	}

	return out, nil
}

func parseGroupLine(line string) (*Group, error) {
	const (
		NameIdx = iota
		PassIdx
		GidIdx
		UsersIdx
	)
	var err error

	if line == "" {
		return nil, errors.New("cannot parse empty line")
	}

	splits := strings.Split(line, ":")
	if len(splits) < 4 {
		return nil, fmt.Errorf("expected at least 4 fields, got %d", len(splits))
	}

	group := &Group{
		Name: splits[NameIdx],
		Pass: splits[PassIdx],
	}

	group.Gid, err = strconv.Atoi(splits[GidIdx])
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to parse gid"), err)
	}

	u := splits[UsersIdx]
	if u != "" {
		group.Users = strings.Split(u, ",")
	} else {
		group.Users = []string{}
	}

	return group, nil
}
