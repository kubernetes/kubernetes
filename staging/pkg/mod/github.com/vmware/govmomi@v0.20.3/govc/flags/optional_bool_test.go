/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package flags

import (
	"flag"
	"testing"
)

func TestOptionalBool(t *testing.T) {
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	var val *bool

	fs.Var(NewOptionalBool(&val), "obool", "optional bool")

	b := fs.Lookup("obool")

	if b.DefValue != "<nil>" {
		t.Fail()
	}

	if b.Value.String() != "<nil>" {
		t.Fail()
	}

	if b.Value.(flag.Getter).Get() != nil {
		t.Fail()
	}

	b.Value.Set("true")

	if b.Value.String() != "true" {
		t.Fail()
	}

	if b.Value.(flag.Getter).Get() != true {
		t.Fail()
	}

	if val == nil || *val != true {
		t.Fail()
	}

	b.Value.Set("false")

	if b.Value.String() != "false" {
		t.Fail()
	}

	if b.Value.(flag.Getter).Get() != false {
		t.Fail()
	}

	if val == nil || *val != false {
		t.Fail()
	}
}
