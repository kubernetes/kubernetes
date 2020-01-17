/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"crypto/subtle"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"k8s.io/klog"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

type PasswordAuthenticator struct {
	users map[string]*userPasswordInfo
}

type userPasswordInfo struct {
	info     *user.DefaultInfo
	password string
}

// NewCSV returns a PasswordAuthenticator, populated from a CSV file.
// The CSV file must contain records in the format "password,username,useruid"
func NewCSV(path string) (*PasswordAuthenticator, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	recordNum := 0
	users := make(map[string]*userPasswordInfo)
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
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
		obj := &userPasswordInfo{
			info:     &user.DefaultInfo{Name: record[1], UID: record[2]},
			password: record[0],
		}
		if len(record) >= 4 {
			obj.info.Groups = strings.Split(record[3], ",")
		}
		recordNum++
		if _, exist := users[obj.info.Name]; exist {
			klog.Warningf("duplicate username '%s' has been found in password file '%s', record number '%d'", obj.info.Name, path, recordNum)
		}
		users[obj.info.Name] = obj
	}

	return &PasswordAuthenticator{users}, nil
}

func (a *PasswordAuthenticator) AuthenticatePassword(ctx context.Context, username, password string) (*authenticator.Response, bool, error) {
	user, ok := a.users[username]
	if !ok {
		return nil, false, nil
	}
	if subtle.ConstantTimeCompare([]byte(user.password), []byte(password)) == 0 {
		return nil, false, nil
	}
	return &authenticator.Response{User: user.info}, true, nil
}
