// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"golang.org/x/oauth2"
)

type urlCredentialSource struct {
	URL     string
	Headers map[string]string
	Format  Format
	ctx     context.Context
}

func (cs urlCredentialSource) credentialSourceType() string {
	return "url"
}

func (cs urlCredentialSource) subjectToken() (string, error) {
	client := oauth2.NewClient(cs.ctx, nil)
	req, err := http.NewRequest("GET", cs.URL, nil)
	if err != nil {
		return "", fmt.Errorf("oauth2/google/externalaccount: HTTP request for URL-sourced credential failed: %v", err)
	}
	req = req.WithContext(cs.ctx)

	for key, val := range cs.Headers {
		req.Header.Add(key, val)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("oauth2/google/externalaccount: invalid response when retrieving subject token: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("oauth2/google/externalaccount: invalid body in subject token URL query: %v", err)
	}
	if c := resp.StatusCode; c < 200 || c > 299 {
		return "", fmt.Errorf("oauth2/google/externalaccount: status code %d: %s", c, respBody)
	}

	switch cs.Format.Type {
	case "json":
		jsonData := make(map[string]any)
		err = json.Unmarshal(respBody, &jsonData)
		if err != nil {
			return "", fmt.Errorf("oauth2/google/externalaccount: failed to unmarshal subject token file: %v", err)
		}
		val, ok := jsonData[cs.Format.SubjectTokenFieldName]
		if !ok {
			return "", errors.New("oauth2/google/externalaccount: provided subject_token_field_name not found in credentials")
		}
		token, ok := val.(string)
		if !ok {
			return "", errors.New("oauth2/google/externalaccount: improperly formatted subject token")
		}
		return token, nil
	case "text":
		return string(respBody), nil
	case "":
		return string(respBody), nil
	default:
		return "", errors.New("oauth2/google/externalaccount: invalid credential_source file format type")
	}

}
