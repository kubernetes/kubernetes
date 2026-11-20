//go:build linux

/*
Copyright The Kubernetes Authors.

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

package fsquota

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteProjectFiles(t *testing.T) {
	for _, tc := range []struct {
		name        string
		writeProjid bool
		failure     string
	}{
		{name: "update both files", writeProjid: true},
		{name: "update projects only"},
		{name: "projid write fails", writeProjid: true, failure: "write"},
		{name: "projid rename fails", writeProjid: true, failure: "rename"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			projectsPath := filepath.Join(dir, "projects")
			projidPath := filepath.Join(dir, "projid")
			for _, path := range []string{projectsPath, projidPath} {
				if err := os.WriteFile(path, []byte("original\n"), 0644); err != nil {
					t.Fatal(err)
				}
			}
			projects, err := os.Open(projectsPath)
			if err != nil {
				t.Fatal(err)
			}
			defer projects.Close()
			projid, err := os.Open(projidPath)
			if err != nil {
				t.Fatal(err)
			}
			defer projid.Close()
			switch tc.failure {
			case "write":
				if err := projid.Close(); err != nil {
					t.Fatal(err)
				}
			case "rename":
				// Replacing the destination with a directory makes rename fail without root or mocks.
				if err := os.Remove(projidPath); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(projidPath, 0755); err != nil {
					t.Fatal(err)
				}
			}
			list := projectsList{
				projects: []projectType{{isValid: true, line: "1048577:/volume"}},
				projid:   []projectType{{isValid: true, line: "volume1048577:1048577"}},
			}
			err = writeProjectFiles(projects, projid, tc.writeProjid, list)
			wantErr := tc.failure != ""
			if (err != nil) != wantErr {
				t.Errorf("writeProjectFiles() error = %v, want error = %v", err, wantErr)
			}
			wantProjects := "1048577:/volume\n"
			if wantErr {
				wantProjects = "original\n"
			}
			if data, err := os.ReadFile(projectsPath); err != nil || string(data) != wantProjects {
				t.Errorf("projects = %q, error = %v; want %q", data, err, wantProjects)
			}
			if tc.failure != "rename" {
				wantProjid := "original\n"
				if tc.writeProjid && !wantErr {
					wantProjid = "volume1048577:1048577\n"
				}
				if data, err := os.ReadFile(projidPath); err != nil || string(data) != wantProjid {
					t.Errorf("projid = %q, error = %v; want %q", data, err, wantProjid)
				}
			}
			if entries, err := os.ReadDir(dir); err != nil || len(entries) != 2 {
				t.Errorf("temporary files remain: entries = %v, error = %v", entries, err)
			}
		})
	}
}
