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
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
)

func jsonFromBody(r io.Reader, v interface{}) error {

	// Check body
	body, err := ioutil.ReadAll(r)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(body, v); err != nil {
		return err
	}

	return nil
}

// Unmarshal JSON from request
func GetJsonFromRequest(r *http.Request, v interface{}) error {
	defer r.Body.Close()
	return jsonFromBody(r.Body, v)
}

// Unmarshal JSON from response
func GetJsonFromResponse(r *http.Response, v interface{}) error {
	defer r.Body.Close()
	return jsonFromBody(r.Body, v)
}
