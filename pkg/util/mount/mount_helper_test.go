/*
Copyright 2018 The Kubernetes Authors.

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

package mount

import (
	"os"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"
)

func TestDoUnmountMountPoint(t *testing.T) {

	tmpDir1, err1 := utiltesting.MkTmpdir("umount_test1")
	if err1 != nil {
		t.Fatalf("error creating temp dir: %v", err1)
	}
	defer os.RemoveAll(tmpDir1)

	tmpDir2, err2 := utiltesting.MkTmpdir("umount_test2")
	if err2 != nil {
		t.Fatalf("error creating temp dir: %v", err2)
	}
	defer os.RemoveAll(tmpDir2)

	// Second part: want no error
	tests := []struct {
		mountPath    string
		corruptedMnt bool
	}{
		{
			mountPath:    tmpDir1,
			corruptedMnt: true,
		},
		{
			mountPath:    tmpDir2,
			corruptedMnt: false,
		},
	}

	fake := &FakeMounter{}

	for _, tt := range tests {
		err := doUnmountMountPoint(tt.mountPath, fake, false, tt.corruptedMnt)
		if err != nil {
			t.Errorf("err Expected nil, but got: %v", err)
		}
	}
}
