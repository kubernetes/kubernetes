//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package utils

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
)

// Return the body from a response as a string
func GetStringFromResponse(r *http.Response) (string, error) {
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	defer r.Body.Close()
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// Return the body from a response as an error
func GetErrorFromResponse(r *http.Response) error {
	s, err := GetStringFromResponse(r)
	if err != nil {
		return err
	}

	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return fmt.Errorf("server did not provide a message (status %v: %v)", r.StatusCode, http.StatusText(r.StatusCode))
	}
	return errors.New(s)
}
