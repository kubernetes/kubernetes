/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"
)

func TestExtractBucketAndFile(t *testing.T) {
	tests := []struct {
		input          string
		expectedBucket string
		expectedFile   string
	}{}
	for _, test := range tests {
		bucket, file := extractBucketAndFile(test.input)
		if bucket != test.expectedBucket {
			t.Errorf("expected: %s, saw: %s", test.expectedBucket, bucket)
		}
		if file != test.expectedFile {
			t.Errorf("expected: %s, saw: %s", test.expectedFile, file)
		}
	}
}
