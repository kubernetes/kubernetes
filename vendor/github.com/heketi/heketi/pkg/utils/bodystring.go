//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), as published by the Free Software Foundation,
// or under the Apache License, Version 2.0 <LICENSE-APACHE2 or
// http://www.apache.org/licenses/LICENSE-2.0>.
//
// You may not use this file except in compliance with those terms.
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

var (
	errMax = int64(4096)
	strMax = int64(8192)
)

// Return the body from a response as a string
func GetStringFromResponse(r *http.Response) (string, error) {
	// If the content length is not set, limit reading to 8K worth of data.
	return getResponse(r, strMax)
}

func getResponse(r *http.Response, max int64) (string, error) {
	if r.ContentLength >= 0 {
		max = r.ContentLength
	}
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, max))
	defer r.Body.Close()
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// Return the body from a response as an error
func GetErrorFromResponse(r *http.Response) error {
	// If the content length is not set, limit reading to 4K worth of data.
	// It is probably way more than needed because an error that long is
	// very unusual. Plus it will only cut it off rather than show nothing.
	s, err := getResponse(r, errMax)
	if err != nil {
		return err
	}

	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return fmt.Errorf("server did not provide a message (status %v: %v)", r.StatusCode, http.StatusText(r.StatusCode))
	}
	return errors.New(s)
}
