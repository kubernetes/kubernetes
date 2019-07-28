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

package hash

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
)

func TestConfigMapHash(t *testing.T) {
	cases := []struct {
		desc string
		cm   *v1.ConfigMap
		hash string
		err  string
	}{
		// empty map
		{"empty data", &v1.ConfigMap{Data: map[string]string{}, BinaryData: map[string][]byte{}}, "42745tchd9", ""},
		// one key
		{"one key", &v1.ConfigMap{Data: map[string]string{"one": ""}}, "9g67k2htb6", ""},
		// three keys (tests sorting order)
		{"three keys", &v1.ConfigMap{Data: map[string]string{"two": "2", "one": "", "three": "3"}}, "f5h7t85m9b", ""},
		// empty binary data map
		{"empty binary data", &v1.ConfigMap{BinaryData: map[string][]byte{}}, "dk855m5d49", ""},
		// one key with binary data
		{"one key with binary data", &v1.ConfigMap{BinaryData: map[string][]byte{"one": []byte("")}}, "mk79584b8c", ""},
		// three keys with binary data (tests sorting order)
		{"three keys with binary data", &v1.ConfigMap{BinaryData: map[string][]byte{"two": []byte("2"), "one": []byte(""), "three": []byte("3")}}, "t458mc6db2", ""},
		// two keys, one with string and another with binary data
		{"two keys with one each", &v1.ConfigMap{Data: map[string]string{"one": ""}, BinaryData: map[string][]byte{"two": []byte("")}}, "698h7c7t9m", ""},
	}

	for _, c := range cases {
		h, err := ConfigMapHash(c.cm)
		if SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if c.hash != h {
			t.Errorf("case %q, expect hash %q but got %q", c.desc, c.hash, h)
		}
	}
}

func TestSecretHash(t *testing.T) {
	cases := []struct {
		desc   string
		secret *v1.Secret
		hash   string
		err    string
	}{
		// empty map
		{"empty data", &v1.Secret{Type: "my-type", Data: map[string][]byte{}}, "t75bgf6ctb", ""},
		// one key
		{"one key", &v1.Secret{Type: "my-type", Data: map[string][]byte{"one": []byte("")}}, "74bd68bm66", ""},
		// three keys (tests sorting order)
		{"three keys", &v1.Secret{Type: "my-type", Data: map[string][]byte{"two": []byte("2"), "one": []byte(""), "three": []byte("3")}}, "dgcb6h9tmk", ""},
	}

	for _, c := range cases {
		h, err := SecretHash(c.secret)
		if SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if c.hash != h {
			t.Errorf("case %q, expect hash %q but got %q", c.desc, c.hash, h)
		}
	}
}

func TestEncodeConfigMap(t *testing.T) {
	cases := []struct {
		desc   string
		cm     *v1.ConfigMap
		expect string
		err    string
	}{
		// empty map
		{"empty data", &v1.ConfigMap{Data: map[string]string{}}, `{"data":{},"kind":"ConfigMap","name":""}`, ""},
		// one key
		{"one key", &v1.ConfigMap{Data: map[string]string{"one": ""}}, `{"data":{"one":""},"kind":"ConfigMap","name":""}`, ""},
		// three keys (tests sorting order)
		{"three keys", &v1.ConfigMap{Data: map[string]string{"two": "2", "one": "", "three": "3"}}, `{"data":{"one":"","three":"3","two":"2"},"kind":"ConfigMap","name":""}`, ""},
		// empty binary map
		{"empty data", &v1.ConfigMap{BinaryData: map[string][]byte{}}, `{"data":null,"kind":"ConfigMap","name":""}`, ""},
		// one key with binary data
		{"one key", &v1.ConfigMap{BinaryData: map[string][]byte{"one": []byte("")}}, `{"binaryData":{"one":""},"data":null,"kind":"ConfigMap","name":""}`, ""},
		// three keys with binary data (tests sorting order)
		{"three keys", &v1.ConfigMap{BinaryData: map[string][]byte{"two": []byte("2"), "one": []byte(""), "three": []byte("3")}}, `{"binaryData":{"one":"","three":"Mw==","two":"Mg=="},"data":null,"kind":"ConfigMap","name":""}`, ""},
		// two keys, one string and one binary values
		{"two keys with one each", &v1.ConfigMap{Data: map[string]string{"one": ""}, BinaryData: map[string][]byte{"two": []byte("")}}, `{"binaryData":{"two":""},"data":{"one":""},"kind":"ConfigMap","name":""}`, ""},
	}
	for _, c := range cases {
		s, err := encodeConfigMap(c.cm)
		if SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if s != c.expect {
			t.Errorf("case %q, expect %q but got %q from encode %#v", c.desc, c.expect, s, c.cm)
		}
	}
}

func TestEncodeSecret(t *testing.T) {
	cases := []struct {
		desc   string
		secret *v1.Secret
		expect string
		err    string
	}{
		// empty map
		{"empty data", &v1.Secret{Type: "my-type", Data: map[string][]byte{}}, `{"data":{},"kind":"Secret","name":"","type":"my-type"}`, ""},
		// one key
		{"one key", &v1.Secret{Type: "my-type", Data: map[string][]byte{"one": []byte("")}}, `{"data":{"one":""},"kind":"Secret","name":"","type":"my-type"}`, ""},
		// three keys (tests sorting order) - note json.Marshal base64 encodes the values because they come in as []byte
		{"three keys", &v1.Secret{Type: "my-type", Data: map[string][]byte{"two": []byte("2"), "one": []byte(""), "three": []byte("3")}}, `{"data":{"one":"","three":"Mw==","two":"Mg=="},"kind":"Secret","name":"","type":"my-type"}`, ""},
	}
	for _, c := range cases {
		s, err := encodeSecret(c.secret)
		if SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if s != c.expect {
			t.Errorf("case %q, expect %q but got %q from encode %#v", c.desc, c.expect, s, c.secret)
		}
	}
}

func TestHash(t *testing.T) {
	// hash the empty string to be sure that sha256 is being used
	expect := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	sum := hash("")
	if expect != sum {
		t.Errorf("expected hash %q but got %q", expect, sum)
	}
}

// warn devs who change types that they might have to update a hash function
// not perfect, as it only checks the number of top-level fields
func TestTypeStability(t *testing.T) {
	errfmt := `case %q, expected %d fields but got %d
Depending on the field(s) you added, you may need to modify the hash function for this type.
To guide you: the hash function targets fields that comprise the contents of objects,
not their metadata (e.g. the Data of a ConfigMap, but nothing in ObjectMeta).
`
	cases := []struct {
		typeName string
		obj      interface{}
		expect   int
	}{
		{"ConfigMap", v1.ConfigMap{}, 4},
		{"Secret", v1.Secret{}, 5},
	}
	for _, c := range cases {
		val := reflect.ValueOf(c.obj)
		if num := val.NumField(); c.expect != num {
			t.Errorf(errfmt, c.typeName, c.expect, num)
		}
	}
}

// SkipRest returns true if there was a non-nil error or if we expected an error that didn't happen,
// and logs the appropriate error on the test object.
// The return value indicates whether we should skip the rest of the test case due to the error result.
func SkipRest(t *testing.T, desc string, err error, contains string) bool {
	if err != nil {
		if len(contains) == 0 {
			t.Errorf("case %q, expect nil error but got %q", desc, err.Error())
		} else if !strings.Contains(err.Error(), contains) {
			t.Errorf("case %q, expect error to contain %q but got %q", desc, contains, err.Error())
		}
		return true
	} else if len(contains) > 0 {
		t.Errorf("case %q, expect error to contain %q but got nil error", desc, contains)
		return true
	}
	return false
}
