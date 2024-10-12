/*
Copyright 2024 The Kubernetes Authors.

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

package dryrun

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestPrintDryRunFiles(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer func() {
		if err = os.RemoveAll(tmpdir); err != nil {
			t.Fatalf("Couldn't remove tmpdir: %v", err)
		}
	}()
	fileContents := "apiVersion: kubeadm.k8s.io/unknownVersion"
	filename := "testfile"
	cfgPath := filepath.Join(tmpdir, filename)
	err = os.WriteFile(cfgPath, []byte(fileContents), 0644)
	if err != nil {
		t.Fatalf("Couldn't write context to file: %v", err)
	}

	tests := []struct {
		name    string
		files   []FileToPrint
		wantW   string
		wantErr bool
	}{
		{
			name: "RealPath is empty",
			files: []FileToPrint{
				{
					RealPath:  "",
					PrintPath: cfgPath,
				},
			},
			wantW:   "",
			wantErr: false,
		},
		{
			name: "RealPath is a file that does not exist",
			files: []FileToPrint{
				{
					RealPath:  tmpdir + "/missingfile",
					PrintPath: cfgPath,
				},
			},
			wantW:   "",
			wantErr: true,
		},
		{
			name: "RealPath is a readable file",
			files: []FileToPrint{
				{
					RealPath:  cfgPath,
					PrintPath: "",
				},
			},
			wantW: "[dryrun] Would write file \"" + cfgPath + "\" with content:\n" +
				"	apiVersion: kubeadm.k8s.io/unknownVersion\n",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &bytes.Buffer{}
			if err := PrintDryRunFiles(tt.files, w); (err != nil) != tt.wantErr {
				t.Errorf("error: %v, expected error: %v", err, tt.wantErr)
				return
			}
			if gotW := w.String(); gotW != tt.wantW {
				t.Errorf("output: %v, expected output: %v", gotW, tt.wantW)
			}
		})
	}
}

func TestNewFileToPrint(t *testing.T) {
	tests := []struct {
		realPath  string
		printPath string
		want      FileToPrint
	}{
		{
			realPath:  "",
			printPath: "",
			want:      FileToPrint{},
		},
		{
			realPath:  "/etc/kubernetes",
			printPath: "/tmp/kubernetes",
			want: FileToPrint{
				"/etc/kubernetes",
				"/tmp/kubernetes",
			},
		},
	}
	for _, tt := range tests {
		t.Run("TestNewFileToPrint", func(t *testing.T) {
			if got := NewFileToPrint(tt.realPath, tt.printPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got: %v, want: %v", got, tt.want)
			}
		})
	}
}
