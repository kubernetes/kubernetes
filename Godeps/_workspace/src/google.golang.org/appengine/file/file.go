// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package file provides helper functions for using Google Cloud Storage.
package file

import (
	"fmt"

	"google.golang.org/appengine"
	aipb "google.golang.org/appengine/internal/app_identity"
)

// DefaultBucketName returns the name of this application's
// default Google Cloud Storage bucket.
func DefaultBucketName(c appengine.Context) (string, error) {
	req := &aipb.GetDefaultGcsBucketNameRequest{}
	res := &aipb.GetDefaultGcsBucketNameResponse{}

	err := c.Call("app_identity_service", "GetDefaultGcsBucketName", req, res, nil)
	if err != nil {
		return "", fmt.Errorf("file: no default bucket name returned in RPC response: %v", res)
	}
	return res.GetDefaultGcsBucketName(), nil
}
