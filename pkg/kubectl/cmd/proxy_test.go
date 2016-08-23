/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/mock"
)

type executorMock struct {
	mock.Mock
}

var _ executorInterface = new(executorMock)

func (m executorMock) Exec(argv0 string, argv, envv []string) error {
	args := m.Called(argv0, argv, envv)
	return args.Error(0)
}

func (m executorMock) LookPath(argv []string) (string, error) {
	return fmt.Sprintf("/usr/bin/%s", argv[0]), nil
}

func TestRunProxyExec(t *testing.T) {
	tests := []struct {
		execCommand                      string
		address                          string
		port                             int
		expectedArgv0                    string
		expectedFormattedSplittedCommand []string
	}{
		{
			execCommand:                      "curl {}/api",
			address:                          "127.0.0.1",
			port:                             8080,
			expectedArgv0:                    "/usr/bin/curl",
			expectedFormattedSplittedCommand: []string{"curl", "http://127.0.0.1:8080/api"},
		},
		{
			execCommand:                      "links {}/api/v1",
			address:                          "10.0.0.1",
			port:                             9091,
			expectedArgv0:                    "/usr/bin/links",
			expectedFormattedSplittedCommand: []string{"links", "http://10.0.0.1:9091/api/v1"},
		},
	}
	m := &executorMock{}
	p := proxyExec{executor: m}
	for _, tc := range tests {
		m.On("Exec", tc.expectedArgv0, tc.expectedFormattedSplittedCommand, os.Environ()).Return(nil)
		err := p.Run(tc.execCommand, tc.address, tc.port)
		if err != nil {
			t.Fatal(err)
		}
	}
}
