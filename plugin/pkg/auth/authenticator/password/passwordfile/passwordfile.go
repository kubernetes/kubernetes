/*
Copyright 2015 Google Inc. All rights reserved.

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

package passwordfile

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
)

type PasswordAuthenticator struct {
	passwords map[string]*user.DefaultInfo
}

// NewCSV returns a PasswordAuthenticator, populated from a CSV file.
// The CSV file must contain records in the format "password,username,useruid"
func NewCSV(path string) (*PasswordAuthenticator, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	passwords := make(map[string]*user.DefaultInfo)
	reader := csv.NewReader(file)
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) < 3 {
			return nil, fmt.Errorf("password file '%s' must have at least 3 columns (password, user name, user uid), found %d", path, len(record))
		}
		obj := &user.DefaultInfo{
			Name: record[1],
			UID:  record[2],
		}
		passwords[record[0]] = obj
	}

	return &PasswordAuthenticator{passwords}, nil
}

func (a *PasswordAuthenticator) AuthenticatePassword(username, password string) (user.Info, bool, error) {
	user, ok := a.passwords[password]
	if !ok {
		return nil, false, nil
	}
	if user.Name != username {
		return nil, false, nil
	}
	return user, true, nil
}
