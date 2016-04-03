// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
helloworld tracks how often a user has visited the index page.

This program demonstrates usage of the Cloud Bigtable API for Managed VMs and Go.
Instructions for running this program are in the README.md.
*/
package main

import (
	"bytes"
	"encoding/binary"
	"html/template"
	"log"
	"net/http"

	"golang.org/x/net/context"
	"google.golang.org/appengine"
	aelog "google.golang.org/appengine/log"
	"google.golang.org/appengine/user"
	"google.golang.org/cloud/bigtable"
)

// User-provided constants.
const (
	project = "PROJECT_ID"
	zone    = "CLUSTER_ZONE"
	cluster = "CLUSTER_NAME"
)

var (
	tableName  = "bigtable-hello"
	familyName = "emails"

	// Client is initialized by main.
	client *bigtable.Client
)

func main() {
	ctx := context.Background()

	// Set up admin client, tables, and column families.
	// NewAdminClient uses Application Default Credentials to authenticate.
	adminClient, err := bigtable.NewAdminClient(ctx, project, zone, cluster)
	if err != nil {
		log.Fatalf("Unable to create a table admin client. %v", err)
	}
	tables, err := adminClient.Tables(ctx)
	if err != nil {
		log.Fatalf("Unable to fetch table list. %v", err)
	}
	if !sliceContains(tables, tableName) {
		if err := adminClient.CreateTable(ctx, tableName); err != nil {
			log.Fatalf("Unable to create table: %v. %v", tableName, err)
		}
	}
	tblInfo, err := adminClient.TableInfo(ctx, tableName)
	if err != nil {
		log.Fatalf("Unable to read info for table: %v. %v", tableName, err)
	}
	if !sliceContains(tblInfo.Families, familyName) {
		if err := adminClient.CreateColumnFamily(ctx, tableName, familyName); err != nil {
			log.Fatalf("Unable to create column family: %v. %v", familyName, err)
		}
	}
	adminClient.Close()

	// Set up Bigtable data operations client.
	// NewClient uses Application Default Credentials to authenticate.
	client, err = bigtable.NewClient(ctx, project, zone, cluster)
	if err != nil {
		log.Fatalf("Unable to create data operations client. %v", err)
	}

	http.Handle("/", appHandler(mainHandler))
	appengine.Main() // Never returns.
}

// mainHandler tracks how many times each user has visited this page.
func mainHandler(w http.ResponseWriter, r *http.Request) *appError {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return nil
	}

	ctx := appengine.NewContext(r)
	u := user.Current(ctx)
	if u == nil {
		login, err := user.LoginURL(ctx, r.URL.String())
		if err != nil {
			return &appError{err, "Error finding login URL", http.StatusInternalServerError}
		}
		http.Redirect(w, r, login, http.StatusFound)
		return nil
	}
	logoutURL, err := user.LogoutURL(ctx, "/")
	if err != nil {
		return &appError{err, "Error finding logout URL", http.StatusInternalServerError}
	}

	// Display hello page.
	tbl := client.Open(tableName)
	rmw := bigtable.NewReadModifyWrite()
	rmw.Increment(familyName, u.Email, 1)
	row, err := tbl.ApplyReadModifyWrite(ctx, u.Email, rmw)
	if err != nil {
		return &appError{err, "Error applying ReadModifyWrite to row: " + u.Email, http.StatusInternalServerError}
	}
	data := struct {
		Username, Logout string
		Visits           uint64
	}{
		Username: u.Email,
		// Retrieve the most recently edited column.
		Visits: binary.BigEndian.Uint64(row[familyName][0].Value),
		Logout: logoutURL,
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return &appError{err, "Error writing template", http.StatusInternalServerError}
	}
	buf.WriteTo(w)
	return nil
}

var tmpl = template.Must(template.New("").Parse(`
<html><body>

<p>
{{with .Username}} Hello {{.}}{{end}}
{{with .Logout}}<a href="{{.}}">Sign out</a>{{end}}

</p>

<p>
You have visited {{.Visits}}
</p>

</body></html>`))

// sliceContains reports whether the provided string is present in the given slice of strings.
func sliceContains(list []string, target string) bool {
	for _, s := range list {
		if s == target {
			return true
		}
	}
	return false
}

// More info about this method of error handling can be found at: http://blog.golang.org/error-handling-and-go
type appHandler func(http.ResponseWriter, *http.Request) *appError

type appError struct {
	Error   error
	Message string
	Code    int
}

func (fn appHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if e := fn(w, r); e != nil {
		ctx := appengine.NewContext(r)
		aelog.Errorf(ctx, "%v", e.Error)
		http.Error(w, e.Message, e.Code)
	}
}
