// Copyright 2015 The appc Authors
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

package types

import (
	"testing"
)

func TestGoodPort(t *testing.T) {
	p := Port{
		Port:  32456,
		Count: 100,
	}
	if err := p.assertValid(); err != nil {
		t.Errorf("good port assertion failed: %v", err)
	}
}

func TestBadPort(t *testing.T) {
	p := Port{
		Port: 88888,
	}
	if p.assertValid() == nil {
		t.Errorf("bad port asserted valid")
	}
}

func TestBadRange(t *testing.T) {
	p := Port{
		Port:  32456,
		Count: 45678,
	}
	if p.assertValid() == nil {
		t.Errorf("bad port range asserted valid")
	}
}
