/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package osutil

import (
	"os"
)

func FileExists(filename string) (bool, error) {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}

// borrowed from ioutil.ReadDir
// ReadDir reads the directory named by dirname and returns
// a list of directory entries, minus those with lstat errors
func ReadDirNoExit(dirname string) ([]os.FileInfo, []error, error) {
	if dirname == "" {
		dirname = "."
	}

	f, err := os.Open(dirname)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	names, err := f.Readdirnames(-1)
	list := make([]os.FileInfo, 0, len(names))
	errs := make([]error, 0, len(names))
	for _, filename := range names {
		fip, lerr := os.Lstat(dirname + "/" + filename)
		if os.IsNotExist(lerr) {
			// File disappeared between readdir + stat.
			// Just treat it as if it didn't exist.
			continue
		}

		list = append(list, fip)
		errs = append(errs, lerr)
	}

	return list, errs, nil
}
