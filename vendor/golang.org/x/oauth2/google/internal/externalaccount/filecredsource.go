// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
)

type fileCredentialSource struct {
	File   string
	Format format
}

func (cs fileCredentialSource) credentialSourceType() string {
	return "file"
}

func (cs fileCredentialSource) subjectToken() (string, error) {
	tokenFile, err := os.Open(cs.File)
	if err != nil {
		return "", fmt.Errorf("oauth2/google: failed to open credential file %q", cs.File)
	}
	defer tokenFile.Close()
	tokenBytes, err := ioutil.ReadAll(io.LimitReader(tokenFile, 1<<20))
	if err != nil {
		return "", fmt.Errorf("oauth2/google: failed to read credential file: %v", err)
	}
	tokenBytes = bytes.TrimSpace(tokenBytes)
	switch cs.Format.Type {
	case "json":
		jsonData := make(map[string]interface{})
		err = json.Unmarshal(tokenBytes, &jsonData)
		if err != nil {
			return "", fmt.Errorf("oauth2/google: failed to unmarshal subject token file: %v", err)
		}
		val, ok := jsonData[cs.Format.SubjectTokenFieldName]
		if !ok {
			return "", errors.New("oauth2/google: provided subject_token_field_name not found in credentials")
		}
		token, ok := val.(string)
		if !ok {
			return "", errors.New("oauth2/google: improperly formatted subject token")
		}
		return token, nil
	case "text":
		return string(tokenBytes), nil
	case "":
		return string(tokenBytes), nil
	default:
		return "", errors.New("oauth2/google: invalid credential_source file format type")
	}

}
