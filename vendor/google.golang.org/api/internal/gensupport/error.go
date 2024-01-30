// Copyright 2022 Google LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"errors"

	"github.com/googleapis/gax-go/v2/apierror"
	"google.golang.org/api/googleapi"
)

// WrapError creates an [apierror.APIError] from err, wraps it in err, and
// returns err. If err is not a [googleapi.Error] (or a
// [google.golang.org/grpc/status.Status]), it returns err without modification.
func WrapError(err error) error {
	var herr *googleapi.Error
	apiError, ok := apierror.ParseError(err, false)
	if ok && errors.As(err, &herr) {
		herr.Wrap(apiError)
	}
	return err
}
