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
	"bytes"
	"encoding/hex"
	"encoding/xml"
	"flag"
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	chk "gopkg.in/check.v1"
)

var (
	overwriteRec bool
	pwd          string
)

func TestMain(m *testing.M) {
	var err error
	flag.BoolVar(&overwriteRec, "ow", false, "Regenerate recordings for testing")
	pwd, err = os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Unable to get current working directory: %v\n", err)
		os.Exit(1)
	}
	exitStatus := m.Run()
	err = fixRecordings()
	if err != nil {
		fmt.Fprintf(os.Stderr, "After test run, fixing recordings failed with error: %v\n", err)
		exitStatus = 1
	}
	os.Exit(exitStatus)
}

func fixRecordings() error {
	err := filepath.Walk(recordingsFolder, func(path string, file os.FileInfo, err error) error {
		if strings.ToLower(filepath.Ext(path)) == ".yaml" {
			recording, err := ioutil.ReadFile(path)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading file '%s': %v", path, err)
			}

			fixedRecording := replaceStorageAccount(string(recording))

			err = ioutil.WriteFile(path, []byte(fixedRecording), 0)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error writing file '%s': %v", path, err)
			}
		}
		return err
	})
	return err
}

func replaceStorageAccount(recording string) string {
	name := os.Getenv("ACCOUNT_NAME")
	if name == "" {
		// do nothing
		return recording
	}

	nameHex := getHex(name)
	dummyHex := getHex(dummyStorageAccount)

	r := strings.NewReplacer(name, dummyStorageAccount,
		nameHex, dummyHex)

	return r.Replace(string(recording))
}

func getHex(input string) string {
	encoded := strings.ToUpper(hex.EncodeToString([]byte(input)))
	formatted := bytes.Buffer{}
	for i := 0; i < len(encoded); i += 2 {
		formatted.WriteString(`\x`)
		formatted.WriteString(encoded[i : i+2])
	}
	return formatted.String()
}

const (
	dummyStorageAccount = "golangrocksonazure"
	dummyMiniStorageKey = "YmFy"
	recordingsFolder    = "recordings"
)

func (s *StorageClientSuite) Test_timeRfc1123Formatted(c *chk.C) {
	now := time.Now().UTC()
	expectedLayout := "Mon, 02 Jan 2006 15:04:05 GMT"
	c.Assert(timeRfc1123Formatted(now), chk.Equals, now.Format(expectedLayout))
}

func (s *StorageClientSuite) Test_mergeParams(c *chk.C) {
	v1 := url.Values{
		"k1": {"v1"},
		"k2": {"v2"}}
	v2 := url.Values{
		"k1": {"v11"},
		"k3": {"v3"}}
	out := mergeParams(v1, v2)
	c.Assert(out.Get("k1"), chk.Equals, "v1")
	c.Assert(out.Get("k2"), chk.Equals, "v2")
	c.Assert(out.Get("k3"), chk.Equals, "v3")
	c.Assert(out["k1"], chk.DeepEquals, []string{"v1", "v11"})
}

func (s *StorageClientSuite) Test_prepareBlockListRequest(c *chk.C) {
	empty := []Block{}
	expected := `<?xml version="1.0" encoding="utf-8"?><BlockList></BlockList>`
	c.Assert(prepareBlockListRequest(empty), chk.DeepEquals, expected)

	blocks := []Block{{"lol", BlockStatusLatest}, {"rofl", BlockStatusUncommitted}}
	expected = `<?xml version="1.0" encoding="utf-8"?><BlockList><Latest>lol</Latest><Uncommitted>rofl</Uncommitted></BlockList>`
	c.Assert(prepareBlockListRequest(blocks), chk.DeepEquals, expected)
}

func (s *StorageClientSuite) Test_xmlUnmarshal(c *chk.C) {
	xml := `<?xml version="1.0" encoding="utf-8"?>
	<Blob>
		<Name>myblob</Name>
	</Blob>`
	var blob Blob
	body := ioutil.NopCloser(strings.NewReader(xml))
	c.Assert(xmlUnmarshal(body, &blob), chk.IsNil)
	c.Assert(blob.Name, chk.Equals, "myblob")
}

func (s *StorageClientSuite) Test_xmlMarshal(c *chk.C) {
	type t struct {
		XMLName xml.Name `xml:"S"`
		Name    string   `xml:"Name"`
	}

	b := t{Name: "myblob"}
	expected := `<S><Name>myblob</Name></S>`
	r, i, err := xmlMarshal(b)
	c.Assert(err, chk.IsNil)
	o, err := ioutil.ReadAll(r)
	c.Assert(err, chk.IsNil)
	out := string(o)
	c.Assert(out, chk.Equals, expected)
	c.Assert(i, chk.Equals, len(expected))
}

func (s *StorageClientSuite) Test_headersFromStruct(c *chk.C) {
	type t struct {
		Header1        string     `header:"HEADER1"`
		Header2        string     `header:"HEADER2"`
		TimePtr        *time.Time `header:"ptr-time-header"`
		TimeHeader     time.Time  `header:"time-header"`
		UintPtr        *uint      `header:"ptr-uint-header"`
		UintHeader     uint       `header:"uint-header"`
		IntPtr         *int       `header:"ptr-int-header"`
		IntHeader      int        `header:"int-header"`
		StringAliasPtr *BlobType  `header:"ptr-string-alias-header"`
		StringAlias    BlobType   `header:"string-alias-header"`
		NilPtr         *time.Time `header:"nil-ptr"`
		EmptyString    string     `header:"empty-string"`
	}

	timeHeader := time.Date(1985, time.February, 23, 10, 0, 0, 0, time.Local)
	uintHeader := uint(15)
	intHeader := 30
	alias := BlobTypeAppend
	h := t{
		Header1:        "value1",
		Header2:        "value2",
		TimePtr:        &timeHeader,
		TimeHeader:     timeHeader,
		UintPtr:        &uintHeader,
		UintHeader:     uintHeader,
		IntPtr:         &intHeader,
		IntHeader:      intHeader,
		StringAliasPtr: &alias,
		StringAlias:    alias,
	}
	expected := map[string]string{
		"HEADER1":                 "value1",
		"HEADER2":                 "value2",
		"ptr-time-header":         "Sat, 23 Feb 1985 10:00:00 GMT",
		"time-header":             "Sat, 23 Feb 1985 10:00:00 GMT",
		"ptr-uint-header":         "15",
		"uint-header":             "15",
		"ptr-int-header":          "30",
		"int-header":              "30",
		"ptr-string-alias-header": "AppendBlob",
		"string-alias-header":     "AppendBlob",
	}

	out := headersFromStruct(h)

	c.Assert(out, chk.DeepEquals, expected)
}
