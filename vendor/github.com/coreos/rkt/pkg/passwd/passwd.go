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

package passwd

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
	passwdFilePath = "/etc/passwd"
)

// User represents an entry in the passwd file.
type User struct {
	Name        string
	Pass        string
	Uid         int
	Gid         int
	Comment     string
	Home        string
	Interpreter string
}

// LookupUidFromFile reads the passwd file specified by passwdFile, and returns the
// uid of the user specified by userName.
func LookupUidFromFile(userName, passwdFile string) (uid int, err error) {
	users, err := parsePasswdFile(passwdFile)
	if err != nil {
		return -1, errwrap.Wrap(fmt.Errorf("error parsing %q file", passwdFile), err)
	}

	user, ok := users[userName]
	if !ok {
		return -1, fmt.Errorf("%q user not found", userName)
	}

	return user.Uid, nil
}

// LookupUid reads the passwd file and returns the uid of the user
// specified by userName.
func LookupUid(userName string) (uid int, err error) {
	return LookupUidFromFile(userName, passwdFilePath)
}

func parsePasswdFile(path string) (user map[string]*User, err error) {
	passwdFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer passwdFile.Close()

	return parseUsers(passwdFile)
}

func parseUsers(r io.Reader) (user map[string]*User, err error) {
	s := bufio.NewScanner(r)
	out := make(map[string]*User)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}

		text := s.Text()
		if text == "" {
			continue
		}

		p, err := parsePasswdLine(text)
		if err != nil {
			return nil, errwrap.Wrap(errors.New("error parsing line"), err)
		}

		out[p.Name] = p
	}

	return out, nil
}

func parsePasswdLine(line string) (*User, error) {
	const (
		NameIdx = iota
		PassIdx
		UidIdx
		GidIdx
		CommentIdx
		HomeIdx
		InterpreterIdx
	)
	var err error

	if line == "" {
		return nil, errors.New("cannot parse empty line")
	}

	splits := strings.Split(line, ":")
	if len(splits) < 7 {
		return nil, fmt.Errorf("expected at least 7 fields, got %d", len(splits))
	}

	user := &User{
		Name:        splits[NameIdx],
		Pass:        splits[PassIdx],
		Comment:     splits[CommentIdx],
		Home:        splits[HomeIdx],
		Interpreter: splits[InterpreterIdx],
	}

	user.Uid, err = strconv.Atoi(splits[UidIdx])
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to parse uid"), err)
	}
	user.Gid, err = strconv.Atoi(splits[GidIdx])
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to parse gid"), err)
	}

	return user, nil
}
