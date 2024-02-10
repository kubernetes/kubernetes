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

package util

import (
	"os"
	"testing"
)

func TestCopyFile(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatal("Failed to create temporary directory")
	}
	defer func() {
		err = os.RemoveAll(tmpdir)
		if err != nil {
			t.Fatal("Failed to remove temporary directory")
		}
	}()
	tmpfile, err := os.CreateTemp(tmpdir, "")
	if err != nil {
		t.Fatalf("Failed to create temporary file")
	}
	restrictedFile, err := os.CreateTemp(tmpdir, "")
	if err != nil {
		t.Fatal("Failed to create temporary restricted file")
	}
	err = restrictedFile.Chmod(0000)
	if err != nil {
		t.Fatal("Failed to change file mode")
	}

	type args struct {
		src  string
		dest string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "src file does not exist",
			args: args{
				src:  "foo",
				dest: tmpdir + "bar",
			},
			wantErr: true,
		},
		{
			name: "src file exists",
			args: args{
				src:  tmpfile.Name(),
				dest: tmpdir + "bar",
			},
			wantErr: false,
		},
		{
			name: "src file cannot be read or written",
			args: args{
				src:  restrictedFile.Name(),
				dest: tmpdir + "bar",
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := CopyFile(tt.args.src, tt.args.dest); (err != nil) != tt.wantErr {
				t.Errorf("CopyFile() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
