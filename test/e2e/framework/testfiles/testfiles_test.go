/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package testfiles

import (
	"embed"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	fooContents = `Hello World
`
	fooPath = "testdata/a/foo.txt"

	notExistsPath = "testdata/b"

	expectedDescription = `The following files are embedded into the test executable:
	testdata/a/foo.txt`
)

//go:embed testdata/a
var testFS embed.FS

func getTestEmbeddedSource() *EmbeddedFileSource {
	return &EmbeddedFileSource{
		EmbeddedFS: testFS,
	}
}

func TestEmbeddedFileSource(t *testing.T) {
	s := getTestEmbeddedSource()

	// read a file which exists and compare the contents
	b, err := s.ReadTestFile(fooPath)

	assert.NoError(t, err)
	assert.Equal(t, fooContents, string(b))

	// read a non-existent file and ensure that the returned value is empty and error is nil
	// Note: this is done so that the next file source can be tried by the caller
	b, err = s.ReadTestFile(notExistsPath)
	assert.NoError(t, err)
	assert.Empty(t, b)

	// describing the test filesystem should list down all files
	assert.Equal(t, expectedDescription, s.DescribeFiles())
}
