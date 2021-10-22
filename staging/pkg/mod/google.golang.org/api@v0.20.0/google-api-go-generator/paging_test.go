// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	crm "google.golang.org/api/cloudresourcemanager/v1"
)

//go:generate -command api go run gen.go docurls.go replacements.go -install -api
//go:generate api cloudresourcemanager:v1

// A handler that mimics paging behavior.
type pageHandler struct {
	param bool // is page token in a query param, or body?
	err   error
}

const nPages = 3

func (h *pageHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	v, err := url.ParseRequestURI(r.URL.RequestURI())
	if err != nil {
		h.err = err
		return
	}

	var pageToken string
	if h.param {
		pts := v.Query()["pageToken"]
		if len(pts) > 0 {
			pageToken = pts[0]
		}
	} else {
		d := json.NewDecoder(r.Body)
		req := struct{ PageToken *string }{&pageToken}
		if err := d.Decode(&req); err != nil {
			h.err = err
			return
		}
	}
	var start int
	if pageToken != "" {
		start, err = strconv.Atoi(pageToken)
		if err != nil {
			h.err = err
			return
		}
	}
	nextPageToken := ""
	if start+1 < nPages {
		nextPageToken = strconv.Itoa(start + 1)
	}
	fmt.Fprintf(w, `{"nextPageToken": %q}`, nextPageToken)
}

func TestPagesParam(t *testing.T) {
	handler := &pageHandler{param: true}
	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{}
	s, err := crm.New(client)
	if err != nil {
		t.Fatal(err)
	}
	s.BasePath = server.URL

	ctx := context.Background()
	c := s.Projects.List()

	countPages := func() int {
		n := 0
		err = c.Pages(ctx, func(*crm.ListProjectsResponse) error {
			n++
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return n
	}

	// Pages traverses through all the pages.
	if got, want := countPages(), nPages; got != want {
		t.Errorf("got %d pages, want %d", got, want)
	}

	// Pages starts wherever the current page token is.
	c.PageToken("1")
	if got, want := countPages(), nPages-1; got != want {
		t.Errorf("got %d pages, want %d", got, want)
	}

	// Pages restores the initial state: we will again visit one fewer
	// page, because the initial page token was reset to "1".
	if got, want := countPages(), nPages-1; got != want {
		t.Errorf("got %d pages, want %d", got, want)
	}

	if handler.err != nil {
		t.Fatal(handler.err)
	}
}

func TestPagesRequestField(t *testing.T) {
	handler := &pageHandler{param: false}
	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{}
	s, err := crm.New(client)
	if err != nil {
		t.Fatal(err)
	}
	s.BasePath = server.URL

	ctx := context.Background()
	c := s.Organizations.Search(&crm.SearchOrganizationsRequest{})

	countPages := func() int {
		n := 0
		err = c.Pages(ctx, func(*crm.SearchOrganizationsResponse) error {
			n++
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return n
	}

	// Pages traverses through all the pages.
	if got, want := countPages(), nPages; got != want {
		t.Errorf("got %d pages, want %d", got, want)
	}

	// Pages starts wherever the current page token is.
	c = s.Organizations.Search(&crm.SearchOrganizationsRequest{PageToken: "1"})
	if got, want := countPages(), nPages-1; got != want {
		t.Errorf("got %d pages, want %d", got, want)
	}

	// Pages restores the initial state: we will again visit one fewer
	// page, because the initial page token was reset to "1".
	if got, want := countPages(), nPages-1; got != want {
		t.Errorf("got %d pages, want %d", got, want)
	}
}
