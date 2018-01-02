package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/base64"
	"net/http"

	chk "gopkg.in/check.v1"
)

type AuthorizationSuite struct{}

var _ = chk.Suite(&AuthorizationSuite{})

func (a *AuthorizationSuite) Test_addAuthorizationHeader(c *chk.C) {
	cli, err := NewBasicClient(dummyStorageAccount, dummyMiniStorageKey)
	c.Assert(err, chk.IsNil)
	cli.UseSharedKeyLite = true
	tableCli := cli.GetTableService()

	headers := map[string]string{
		"Accept-Charset":    "UTF-8",
		headerContentType:   "application/json",
		headerXmsDate:       "Wed, 23 Sep 2015 16:40:05 GMT",
		headerContentLength: "0",
		headerXmsVersion:    "2015-02-21",
		"Accept":            "application/json;odata=nometadata",
	}
	url := "https://golangrocksonazure.table.core.windows.net/tquery()"
	headers, err = tableCli.client.addAuthorizationHeader("", url, headers, tableCli.auth)
	c.Assert(err, chk.IsNil)

	c.Assert(headers[headerAuthorization], chk.Equals, "SharedKeyLite golangrocksonazure:NusXSFXAvHqr6EQNXnZZ50CvU1sX0iP/FFDHehnixLc=")
}

func (a *AuthorizationSuite) Test_getSharedKey(c *chk.C) {
	// Shared Key Lite for Tables
	cli, err := NewBasicClient(dummyStorageAccount, dummyMiniStorageKey)
	c.Assert(err, chk.IsNil)

	headers := map[string]string{
		"Accept-Charset":    "UTF-8",
		headerContentType:   "application/json",
		headerXmsDate:       "Wed, 23 Sep 2015 16:40:05 GMT",
		headerContentLength: "0",
		headerXmsVersion:    "2015-02-21",
		"Accept":            "application/json;odata=nometadata",
	}
	url := "https://golangrocksonazure.table.core.windows.net/tquery()"

	key, err := cli.getSharedKey("", url, headers, sharedKeyLiteForTable)
	c.Assert(err, chk.IsNil)
	c.Assert(key, chk.Equals, "SharedKeyLite golangrocksonazure:NusXSFXAvHqr6EQNXnZZ50CvU1sX0iP/FFDHehnixLc=")
}

func (a *AuthorizationSuite) Test_buildCanonicalizedResource(c *chk.C) {
	cli, err := NewBasicClient(dummyStorageAccount, dummyMiniStorageKey)
	c.Assert(err, chk.IsNil)

	type test struct {
		url      string
		auth     authentication
		expected string
		sas      bool
	}
	tests := []test{
		// Shared Key
		{"https://golangrocksonazure.blob.core.windows.net/path?a=b&c=d", sharedKey, "/golangrocksonazure/path\na:b\nc:d", false},
		{"https://golangrocksonazure.blob.core.windows.net/?comp=list", sharedKey, "/golangrocksonazure/\ncomp:list", false},
		{"https://golangrocksonazure.blob.core.windows.net/cnt/blob", sharedKey, "/golangrocksonazure/cnt/blob", false},
		{"https://golangrocksonazure.blob.core.windows.net/cnt/bl ob", sharedKey, "/golangrocksonazure/cnt/bl%20ob", false},
		{"https://golangrocksonazure.blob.core.windows.net/c nt/blob", sharedKey, "/golangrocksonazure/c%20nt/blob", false},
		{"https://golangrocksonazure.blob.core.windows.net/cnt/blob%3F%23%5B%5D%21$&%27%28%29%2A blob", sharedKey, "/golangrocksonazure/cnt/blob%3F%23%5B%5D%21$&%27%28%29%2A%20blob", false},
		{"https://golangrocksonazure.blob.core.windows.net/cnt/blob-._~:,@;+=blob", sharedKey, "/golangrocksonazure/cnt/blob-._~:,@;+=blob", false},
		{"https://golangrocksonazure.blob.core.windows.net/c nt/blob-._~:%3F%23%5B%5D@%21$&%27%28%29%2A,;+=/blob", sharedKey, "/golangrocksonazure/c%20nt/blob-._~:%3F%23%5B%5D@%21$&%27%28%29%2A,;+=/blob", false},
		// Shared Key Lite for Table
		{"https://golangrocksonazure.table.core.windows.net/mytable", sharedKeyLiteForTable, "/golangrocksonazure/mytable", false},
		{"https://golangrocksonazure.table.core.windows.net/mytable?comp=acl", sharedKeyLiteForTable, "/golangrocksonazure/mytable?comp=acl", false},
		{"https://golangrocksonazure.table.core.windows.net/mytable?comp=acl&timeout=10", sharedKeyForTable, "/golangrocksonazure/mytable?comp=acl", false},
		{"https://golangrocksonazure.table.core.windows.net/mytable(PartitionKey='pkey',RowKey='rowkey%3D')", sharedKeyForTable, "/golangrocksonazure/mytable(PartitionKey='pkey',RowKey='rowkey%3D')", false},
		// Canonicalized Resource with SAS
		{"https://golangrocksonazure.blob.core.windows.net/cnt/blob", sharedKey, "/golangrocksonazure/cnt/blob", true},
	}

	for _, t := range tests {
		out, err := cli.buildCanonicalizedResource(t.url, t.auth, t.sas)
		c.Assert(err, chk.IsNil)
		c.Assert(out, chk.Equals, t.expected)
	}

	eCli, err := NewEmulatorClient()
	c.Assert(err, chk.IsNil)
	eTests := []test{
		{"http://127.0.0.1:10000/devstoreaccount1/cnt/blob", sharedKey, "/devstoreaccount1/cnt/blob", true},
		{"http://127.0.0.1:10000/devstoreaccount1/cnt/blob", sharedKey, "/devstoreaccount1/devstoreaccount1/cnt/blob", false},
	}
	for _, t := range eTests {
		out, err := eCli.buildCanonicalizedResource(t.url, t.auth, t.sas)
		c.Assert(err, chk.IsNil)
		c.Assert(out, chk.Equals, t.expected)
	}
}

func (a *AuthorizationSuite) Test_buildCanonicalizedString(c *chk.C) {
	var tests = []struct {
		verb                  string
		headers               map[string]string
		canonicalizedResource string
		auth                  authentication
		out                   string
	}{
		{
			// Shared Key
			verb: http.MethodGet,
			headers: map[string]string{
				headerXmsDate:    "Sun, 11 Oct 2009 21:49:13 GMT",
				headerXmsVersion: "2009-09-19",
			},
			canonicalizedResource: "/myaccount/ mycontainer\ncomp:metadata\nrestype:container\ntimeout:20",
			auth: sharedKey,
			out:  "GET\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:Sun, 11 Oct 2009 21:49:13 GMT\nx-ms-version:2009-09-19\n/myaccount/ mycontainer\ncomp:metadata\nrestype:container\ntimeout:20",
		},
		{
			// Shared Key for Tables
			verb: http.MethodPut,
			headers: map[string]string{
				headerContentType: "text/plain; charset=UTF-8",
				headerDate:        "Sun, 11 Oct 2009 19:52:39 GMT",
			},
			canonicalizedResource: "/testaccount1/Tables",
			auth: sharedKeyForTable,
			out:  "PUT\n\ntext/plain; charset=UTF-8\nSun, 11 Oct 2009 19:52:39 GMT\n/testaccount1/Tables",
		},
		{
			// Shared Key Lite
			verb: http.MethodPut,
			headers: map[string]string{
				headerContentType: "text/plain; charset=UTF-8",
				headerXmsDate:     "Sun, 20 Sep 2009 20:36:40 GMT",
				"x-ms-meta-m1":    "v1",
				"x-ms-meta-m2":    "v2",
			},
			canonicalizedResource: "/testaccount1/mycontainer/hello.txt",
			auth: sharedKeyLite,
			out:  "PUT\n\ntext/plain; charset=UTF-8\n\nx-ms-date:Sun, 20 Sep 2009 20:36:40 GMT\nx-ms-meta-m1:v1\nx-ms-meta-m2:v2\n/testaccount1/mycontainer/hello.txt",
		},
		{
			// Shared Key Lite for Tables
			verb: "",
			headers: map[string]string{
				headerDate: "Sun, 11 Oct 2009 19:52:39 GMT",
			},
			canonicalizedResource: "/testaccount1/Tables",
			auth: sharedKeyLiteForTable,
			out:  "Sun, 11 Oct 2009 19:52:39 GMT\n/testaccount1/Tables",
		},
	}

	for _, t := range tests {
		canonicalizedString, err := buildCanonicalizedString(t.verb, t.headers, t.canonicalizedResource, t.auth)
		c.Assert(err, chk.IsNil)
		c.Assert(canonicalizedString, chk.Equals, t.out)
	}
}

func (a *AuthorizationSuite) Test_buildCanonicalizedHeader(c *chk.C) {
	type test struct {
		headers  map[string]string
		expected string
	}
	tests := []test{
		{map[string]string{},
			""},
		{map[string]string{
			"x-ms-lol": "rofl"},
			"x-ms-lol:rofl"},
		{map[string]string{
			"lol:": "rofl"},
			""},
		{map[string]string{
			"lol:":     "rofl",
			"x-ms-lol": "rofl"},
			"x-ms-lol:rofl"},
		{map[string]string{
			"x-ms-version":   "9999-99-99",
			"x-ms-blob-type": "BlockBlob"},
			"x-ms-blob-type:BlockBlob\nx-ms-version:9999-99-99"}}

	for _, i := range tests {
		c.Assert(buildCanonicalizedHeader(i.headers), chk.Equals, i.expected)
	}
}

func (a *AuthorizationSuite) Test_createAuthorizationHeader(c *chk.C) {
	cli, err := NewBasicClient(dummyStorageAccount, base64.StdEncoding.EncodeToString([]byte("bar")))
	c.Assert(err, chk.IsNil)

	canonicalizedString := `foobarzoo`

	c.Assert(cli.createAuthorizationHeader(canonicalizedString, sharedKey),
		chk.Equals, `SharedKey golangrocksonazure:h5U0ATVX6SpbFX1H6GNuxIMeXXCILLoIvhflPtuQZ30=`)
	c.Assert(cli.createAuthorizationHeader(canonicalizedString, sharedKeyLite),
		chk.Equals, `SharedKeyLite golangrocksonazure:h5U0ATVX6SpbFX1H6GNuxIMeXXCILLoIvhflPtuQZ30=`)
}

func (a *AuthorizationSuite) Test_allSharedKeys(c *chk.C) {
	cli := getBasicClient(c)
	rec := cli.appendRecorder(c)
	defer rec.Stop()

	blobCli := cli.GetBlobService()
	tableCli := cli.GetTableService()

	cnt1 := blobCli.GetContainerReference(containerName(c, "1"))
	cnt2 := blobCli.GetContainerReference(containerName(c, "2"))

	// Shared Key
	c.Assert(blobCli.auth, chk.Equals, sharedKey)
	c.Assert(cnt1.Create(nil), chk.IsNil)
	c.Assert(cnt1.Delete(nil), chk.IsNil)

	// Shared Key for Tables
	c.Assert(tableCli.auth, chk.Equals, sharedKeyForTable)
	table1 := tableCli.GetTableReference(tableName(c, "1"))
	c.Assert(table1.tsc.auth, chk.Equals, sharedKeyForTable)
	c.Assert(table1.Create(30, EmptyPayload, nil), chk.IsNil)
	c.Assert(table1.Delete(30, nil), chk.IsNil)

	// Change to Lite
	cli.UseSharedKeyLite = true
	blobCli = cli.GetBlobService()
	tableCli = cli.GetTableService()

	// Shared Key Lite
	c.Assert(blobCli.auth, chk.Equals, sharedKeyLite)
	c.Assert(cnt2.Create(nil), chk.IsNil)
	c.Assert(cnt2.Delete(nil), chk.IsNil)

	// Shared Key Lite for Tables
	tableCli = cli.GetTableService()
	c.Assert(tableCli.auth, chk.Equals, sharedKeyLiteForTable)
	table2 := tableCli.GetTableReference(tableName(c, "2"))
	c.Assert(table2.tsc.auth, chk.Equals, sharedKeyLiteForTable)
	c.Assert(table2.Create(30, EmptyPayload, nil), chk.IsNil)
	c.Assert(table2.Delete(30, nil), chk.IsNil)
}
