/*
Copyright 2019 The Kubernetes Authors.

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

package routes

import (
	"bufio"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"testing"

	"github.com/emicklei/go-restful"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLogsFollowScanner(t *testing.T) {
	logdir = os.TempDir()
	target, err := ioutil.TempFile(logdir, "logs-follow")
	require.NoError(t, err)
	tname := target.Name()
	defer os.Remove(tname)

	// Setup test server.
	container := restful.NewContainer()
	Logs{}.Install(container)
	server := httptest.NewServer(container)
	defer server.Close()

	resp, err := http.Get(server.URL + path.Join("/logs", filepath.Base(target.Name())) + "?follow=true")
	require.NoError(t, err)
	defer resp.Body.Close()

	testcases := []struct {
		write  string
		expect []string
	}{
		{"first\n", []string{"first"}},
		{"", nil},
		{"second\n", []string{"second"}},
		{"third\nfourth\n", []string{"third", "fourth"}},
	}

	scanner := bufio.NewScanner(resp.Body)

	for _, test := range testcases {
		if test.write != "" {
			_, err = target.WriteString(test.write)
			require.NoError(t, err)
		}

		for _, expect := range test.expect {
			require.True(t, scanner.Scan())
			line := scanner.Text()
			assert.EqualValues(t, expect, line)
		}
	}

	// "rotate" test file
	require.NoError(t, target.Close())
	require.NoError(t, os.Remove(tname))
	target, err = os.OpenFile(tname, os.O_CREATE|os.O_WRONLY, 0644)
	require.NoError(t, err)

	test := "fifth\nsixth\n"
	_, err = target.WriteString(test)
	require.NoError(t, err)

	for _, expect := range []string{"fifth", "sixth"} {
		require.True(t, scanner.Scan())
		line := scanner.Text()
		assert.EqualValues(t, expect, line)
	}
}
