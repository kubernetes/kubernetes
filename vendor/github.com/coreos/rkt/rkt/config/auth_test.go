// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package config

import (
	"testing"
)

func TestGuessAWSRegion(t *testing.T) {
	tests := []struct {
		host   string
		region string
	}{
		{"foo.s3.amazonaws.com", "us-east-1"},
		{"foo.s3-external-1.amazonaws.com", "us-east-1"},
		{"foo.bar.s3.amazonaws.com", "us-east-1"},
		{"foo.s3-us-west-1.amazonaws.com", "us-west-1"},
		{"foo.bar.baz.s3-us-west-2.amazonaws.com", "us-west-2"},
		{"foo.s3-eu-west-1.amazonaws.com", "eu-west-1"},
		{"foo.s3.eu-central-1.amazonaws.com", "eu-central-1"},
		{"foo.bar.s3.eu-central-1.amazonaws.com", "eu-central-1"},
		{"foo.s3-eu-central-1.amazonaws.com", "eu-central-1"},
		{"foo.s3-ap-northeast-1.amazonaws.com", "ap-northeast-1"},
		{"foo.s3.ap-northeast-2.amazonaws.com", "ap-northeast-2"},
		{"foo.s3-ap-northeast-2.amazonaws.com", "ap-northeast-2"},
		{"foo.bar.s3-ap-northeast-2.amazonaws.com", "ap-northeast-2"},
		{"foo.s3-ap-southeast-1.amazonaws.com", "ap-southeast-1"},
		{"foo.bar.baz.s3-ap-southeast-2.amazonaws.com", "ap-southeast-2"},
		{"foo.s3-sa-east-1.amazonaws.com", "sa-east-1"},
		{"foo.s3-ap-southeast-1.amazonaws.com:443", "ap-southeast-1"},
		{"foo.bar.baz.s3-us-west-2.amazonaws.com:80", "us-west-2"},
		{"foo.s3.eu-central-1.amazonaws.com:443", "eu-central-1"},
		{"entirely.unrecognized.url", defaultAWSRegion},
	}

	for _, tt := range tests {
		region := guessAWSRegion(tt.host)
		if region != tt.region {
			t.Errorf("Got unexpected result for %s: %s (expected: %s)", tt.host, region, tt.region)
		}
	}
}
