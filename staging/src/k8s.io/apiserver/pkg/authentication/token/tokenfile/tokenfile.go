/*
Copyright 2014 The Kubernetes Authors.

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

package tokenfile

import (
	"context"
	"crypto/subtle"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/klog/v2"
)

type TokenAuthenticator struct {
	tokens map[string]*user.DefaultInfo
}

// New returns a TokenAuthenticator for a single token
func New(tokens map[string]*user.DefaultInfo) *TokenAuthenticator {
	return &TokenAuthenticator{
		tokens: tokens,
	}
}

// NewCSV returns a TokenAuthenticator, populated from a CSV file.
// The CSV file must contain records in the format "token,username,useruid"
func NewCSV(path string) (*TokenAuthenticator, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	recordNum := 0
	tokens := make(map[string]*user.DefaultInfo)
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
			return nil, fmt.Errorf("token file '%s' must have at least 3 columns (token, user name, user uid), found %d", path, len(record))
		}

		recordNum++
		if record[0] == "" {
			klog.Warningf("empty token has been found in token file '%s', record number '%d'", path, recordNum)
			continue
		}

		obj := &user.DefaultInfo{
			Name: record[1],
			UID:  record[2],
		}
		if _, exist := tokens[record[0]]; exist {
			klog.Warningf("duplicate token has been found in token file '%s', record number '%d'", path, recordNum)
		}
		tokens[record[0]] = obj

		if len(record) >= 4 {
			obj.Groups = strings.Split(record[3], ",")
		}
	}

	return &TokenAuthenticator{
		tokens: tokens,
	}, nil
}

func (a *TokenAuthenticator) AuthenticateToken(ctx context.Context, value string) (*authenticator.Response, bool, error) {
	// Linear scan with crypto/subtle.ConstantTimeCompare to avoid the
	// byte-by-byte timing oracle on the map-lookup memcmp step at the bucket.
	// The sibling bootstrap token authenticator (plugin/pkg/auth/authenticator/
	// token/bootstrap/bootstrap.go) uses the same idiom for its token-vs-secret
	// compare. Static token files are typically small (the feature is deprecated
	// per KEP-2799), so the O(N) scan cost is acceptable.
	valueBytes := []byte(value)
	for storedToken, info := range a.tokens {
		if subtle.ConstantTimeCompare([]byte(storedToken), valueBytes) == 1 {
			return &authenticator.Response{User: info}, true, nil
		}
	}
	return nil, false, nil
}
