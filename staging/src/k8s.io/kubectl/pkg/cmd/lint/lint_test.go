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

package lint

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestNewCmdLint(t *testing.T) {
	var resources []string
	var testnames []string
	err := filepath.Walk("./testdata", func(path string, info os.FileInfo, err error) error {
		if info.Name() == "test.results" {
			path = filepath.Dir(path)
			resources = append(resources, path)
			testnames = append(testnames, filepath.Base(path))
		}
		return nil
	})
	require.NoError(t, err)
	for i := 0; i < len(resources); i++ {
		t.Run(testnames[i], func(t *testing.T) {
			evalTestResults(resources[i], t)
		})
	}
}

func run(filename, want string, t *testing.T) string {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdLint(tf, streams)

	err := cmd.Flags().Set("filename", filename)
	require.NoError(t, err)
	cmd.SetOutput(buf)

	if len(want) > 0 {
		require.Error(t, cmd.RunE(cmd, nil))
	} else {
		require.NoError(t, cmd.RunE(cmd, nil))
	}
	fmt.Println(strings.TrimSpace(buf.String()))
	return strings.TrimSpace(buf.String())
}

func evalTestResults(path string, t *testing.T) {
	var want string
	var testFilePath, testResultPath string
	err := filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		switch filepath.Ext(path) {
		case ".yaml":
			testFilePath = path
		case ".results":
			testResultPath = path
		}
		return nil
	})
	require.NoError(t, err)
	b, err := ioutil.ReadFile(testResultPath)
	require.NoError(t, err)
	want = strings.TrimSpace(string(b))
	require.Equal(t, run(testFilePath, want, t), want)
}
