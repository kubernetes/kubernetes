// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compiler

import (
	"io"
	"net/http"
	"testing"

	"gopkg.in/check.v1"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) {
	check.TestingT(t)
}

var mockServer *http.Server

type ReaderTestingSuite struct{}

var _ = check.Suite(&ReaderTestingSuite{})

func (s *ReaderTestingSuite) SetUpSuite(c *check.C) {
	// prefetch to avoid deadlocking in concurrent calls to ReadBytesForFile
	yamlBytes, err := ReadBytesForFile("testdata/petstore.yaml")
	c.Assert(err, check.IsNil)
	mockServer = &http.Server{Addr: "127.0.0.1:8080", Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, string(yamlBytes))
	})}
	go func() {
		mockServer.ListenAndServe()
	}()
}

func (s *ReaderTestingSuite) TearDownSuite(c *check.C) {
	mockServer.Close()
}

func (s *ReaderTestingSuite) TestRemoveFromInfoCache(c *check.C) {
	fileName := "testdata/petstore.yaml"
	yamlBytes, err := ReadBytesForFile(fileName)
	c.Assert(err, check.IsNil)
	c.Assert(len(yamlBytes) > 0, check.Equals, true)
	petstore, err := ReadInfoFromBytes(fileName, yamlBytes)
	c.Assert(err, check.IsNil)
	c.Assert(petstore, check.NotNil)
	c.Assert(len(infoCache), check.Equals, 1)
	RemoveFromInfoCache(fileName)
	c.Assert(len(infoCache), check.Equals, 0)
}

func (s *ReaderTestingSuite) TestDisableInfoCache(c *check.C) {
	fileName := "testdata/petstore.yaml"
	yamlBytes, err := ReadBytesForFile(fileName)
	c.Assert(err, check.IsNil)
	c.Assert(len(yamlBytes) > 0, check.Equals, true)
	DisableInfoCache()
	petstore, err := ReadInfoFromBytes(fileName, yamlBytes)
	c.Assert(err, check.IsNil)
	c.Assert(petstore, check.NotNil)
	c.Assert(len(infoCache), check.Equals, 0)
	EnableInfoCache()
}

func (s *ReaderTestingSuite) TestRemoveFromFileCache(c *check.C) {
	fileUrl := "http://127.0.0.1:8080/petstore"
	yamlBytes, err := FetchFile(fileUrl)
	c.Assert(err, check.IsNil)
	c.Assert(len(yamlBytes) > 0, check.Equals, true)
	c.Assert(len(fileCache), check.Equals, 1)
	RemoveFromFileCache(fileUrl)
	c.Assert(len(fileCache), check.Equals, 0)
}

func (s *ReaderTestingSuite) TestDisableFileCache(c *check.C) {
	DisableFileCache()
	fileUrl := "http://127.0.0.1:8080/petstore"
	yamlBytes, err := FetchFile(fileUrl)
	c.Assert(err, check.IsNil)
	c.Assert(len(yamlBytes) > 0, check.Equals, true)
	c.Assert(len(fileCache), check.Equals, 0)
	EnableFileCache()
}
