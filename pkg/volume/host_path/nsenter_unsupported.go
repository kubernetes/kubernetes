// +build !linux

/*
Copyright 2017 The Kubernetes Authors.

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

package host_path

type nsenterFileTypeChecker struct {
	path   string
	exists bool
}

func newNsenterFileTypeChecker(path string) (hostPathTypeChecker, error) {
	ftc := &nsenterFileTypeChecker{path: path}
	ftc.Exists()
	return ftc, nil
}

func (ftc *nsenterFileTypeChecker) Exists() bool {
	return false
}

func (ftc *nsenterFileTypeChecker) IsFile() bool {
	return false
}

func (ftc *nsenterFileTypeChecker) MakeFile() error {
	return nil
}

func (ftc *nsenterFileTypeChecker) IsDir() bool {
	return false
}

func (ftc *nsenterFileTypeChecker) MakeDir() error {
	return nil
}

func (ftc *nsenterFileTypeChecker) IsBlock() bool {
	return false
}

func (ftc *nsenterFileTypeChecker) IsChar() bool {
	return false
}

func (ftc *nsenterFileTypeChecker) IsSocket() bool {
	return false
}

func (ftc *nsenterFileTypeChecker) GetPath() string {
	return ftc.path
}
