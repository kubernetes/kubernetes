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

package password

import (
	"regexp"
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

type req struct {
	length int
	chars  string
}

var reqs = []req{
	{},
	{length: 4},
	{chars: "abcd"},
	{length: 8, chars: "1234"},
}

func TestGenerate(t *testing.T) {
	generator := New(nil)

	for _, r := range reqs {
		l := ""
		if r.length > 0 {
			l = strconv.Itoa(r.length)
		}
		genReq := api.GenerateSecretRequest{
			ObjectMeta: api.ObjectMeta{
				Annotations: map[string]string{
					LengthAnnotation: l,
					CharsAnnotation:  r.chars,
				},
			},
		}
		vals, err := generator.GenerateValues(&genReq)
		if err != nil {
			t.Errorf("Unexpected error returned from secret generator: %v", err)
		}
		if len(vals) != 1 {
			t.Errorf("Wrong number of generated values")
		}
		reqLen := r.length
		if reqLen == 0 {
			reqLen = DefaultLength
		}
		password := vals[GeneratedPasswordKey]
		validChars := r.chars
		if len(validChars) == 0 {
			validChars = ASCII
		}
		passwordRE := regexp.MustCompile("^[" + regexp.QuoteMeta(validChars) + "]{" + strconv.Itoa(reqLen) + "}$")
		if !passwordRE.Match(password) {
			t.Errorf("Invalid generated password '%s': should have matched %s", password, passwordRE.String())
		}

	}
}
