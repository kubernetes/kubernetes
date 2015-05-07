// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestLicensesService_List(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/licenses", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testHeader(t, r, "Accept", mediaTypeLicensesPreview)
		fmt.Fprint(w, `[{"key":"mit","name":"MIT","url":"https://api.github.com/licenses/mit"}]`)
	})

	licenses, _, err := client.Licenses.List()
	if err != nil {
		t.Errorf("Licenses.List returned error: %v", err)
	}

	want := []License{License{
		Key:  String("mit"),
		Name: String("MIT"),
		URL:  String("https://api.github.com/licenses/mit"),
	}}
	if !reflect.DeepEqual(licenses, want) {
		t.Errorf("Licenses.List returned %+v, want %+v", licenses, want)
	}
}

func TestLicensesService_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/licenses/mit", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testHeader(t, r, "Accept", mediaTypeLicensesPreview)
		fmt.Fprint(w, `{"key":"mit","name":"MIT"}`)
	})

	license, _, err := client.Licenses.Get("mit")
	if err != nil {
		t.Errorf("Licenses.Get returned error: %v", err)
	}

	want := &License{Key: String("mit"), Name: String("MIT")}
	if !reflect.DeepEqual(license, want) {
		t.Errorf("Licenses.Get returned %+v, want %+v", license, want)
	}
}

func TestLicensesService_Get_invalidTemplate(t *testing.T) {
	_, _, err := client.Licenses.Get("%")
	testURLParseError(t, err)
}
