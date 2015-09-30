/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package v1

import (
	"testing"
)

func tsliceString(t *testing.T, expected string, enc string) {
	var out Buffer
	ffr := newffReader([]byte(enc + `"`))
	err := ffr.SliceString(&out)
	if err != nil {
		t.Fatalf("unexpect SliceString error: %v from %v", err, enc)
	}

	if out.String() != expected {
		t.Fatalf(`failed to decode %v into %v, got: %v`, enc, expected, out.String())
	}
}

func TestUnicode(t *testing.T) {
	var testvecs = map[string]string{
		"€": `\u20AC`,
		"𐐷": `\uD801\uDC37`,
	}

	for k, v := range testvecs {
		tsliceString(t, k, v)
	}
}

func TestBadUnicode(t *testing.T) {
	var out Buffer
	ffr := newffReader([]byte(`\u20--"`))
	err := ffr.SliceString(&out)
	if err == nil {
		t.Fatalf("expected SliceString hex decode error")
	}
}

func TestNonUnicodeEscape(t *testing.T) {
	var out Buffer
	ffr := newffReader([]byte(`\t\n\r"`))
	err := ffr.SliceString(&out)
	if err != nil {
		t.Fatalf("unexpected SliceString error: %v", err)
	}
}

func TestInvalidEscape(t *testing.T) {
	var out Buffer
	ffr := newffReader([]byte(`\x134"`))
	err := ffr.SliceString(&out)
	if err == nil {
		t.Fatalf("expected SliceString escape decode error")
	}
}
