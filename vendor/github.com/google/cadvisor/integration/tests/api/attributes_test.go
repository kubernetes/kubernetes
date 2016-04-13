// Copyright 2015 Google Inc. All Rights Reserved.
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

package api

import (
	"testing"

	"github.com/google/cadvisor/integration/framework"

	"github.com/stretchr/testify/assert"
)

func TestAttributeInformationIsReturned(t *testing.T) {
	fm := framework.New(t)
	defer fm.Cleanup()

	attributes, err := fm.Cadvisor().ClientV2().Attributes()
	if err != nil {
		t.Fatal(err)
	}

	vp := `\d+\.\d+\.\d+`
	assert.True(t, assert.Regexp(t, vp, attributes.DockerVersion),
		"Expected %s to match %s", attributes.DockerVersion, vp)
}
