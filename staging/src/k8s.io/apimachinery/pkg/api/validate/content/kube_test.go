/*
Copyright 2025 The Kubernetes Authors.

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

package content

import (
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestIsLabelKey(t *testing.T) {
	successCases := []string{
		"simple",
		"now-with-dashes",
		"1-starts-with-num",
		"1234",
		"simple/simple",
		"now-with-dashes/simple",
		"now-with-dashes/now-with-dashes",
		"now.with.dots/simple",
		"now-with.dashes-and.dots/simple",
		"1-num.2-num/3-num",
		"1234/5678",
		"1.2.3.4/5678",
		"Uppercase_Is_OK_123",
		"example.com/Uppercase_Is_OK_123",
		"requests.storage-foo",
		strings.Repeat("a", 63),
		strings.Repeat("a", 253) + "/" + strings.Repeat("b", 63),
	}
	for i := range successCases {
		if errs := IsLabelKey(successCases[i]); len(errs) != 0 {
			t.Errorf("case[%d]: %q: expected success: %v", i, successCases[i], errs)
		}
	}

	errorCases := []string{
		"nospecialchars%^=@",
		"cantendwithadash-",
		"-cantstartwithadash-",
		"only/one/slash",
		"Example.com/abc",
		"example_com/abc",
		"example.com/",
		"/simple",
		strings.Repeat("a", 64),
		strings.Repeat("a", 254) + "/abc",
	}
	for i := range errorCases {
		if errs := IsLabelKey(errorCases[i]); len(errs) == 0 {
			t.Errorf("case[%d]: %q: expected failure", i, errorCases[i])
		}
	}
}

func TestIsLabelValue(t *testing.T) {
	successCases := []string{
		"simple",
		"now-with-dashes",
		"1-starts-with-num",
		"end-with-num-1",
		"1234",                  // only num
		strings.Repeat("a", 63), // to the limit
		"",                      // empty value
	}
	for i := range successCases {
		if errs := IsLabelValue(successCases[i]); len(errs) != 0 {
			t.Errorf("case %s expected success: %v", successCases[i], errs)
		}
	}

	errorCases := []string{
		"nospecialchars%^=@",
		"Tama-nui-te-rā.is.Māori.sun",
		"\\backslashes\\are\\bad",
		"-starts-with-dash",
		"ends-with-dash-",
		".starts.with.dot",
		"ends.with.dot.",
		strings.Repeat("a", 64), // over the limit
	}
	for i := range errorCases {
		if errs := IsLabelValue(errorCases[i]); len(errs) == 0 {
			t.Errorf("case[%d] expected failure", i)
		}
	}
}

func TestIsPrefixedLabelKey(t *testing.T) {
	successCases := []string{
		"simple/simple",
		"now-with-dashes/simple",
		"now-with-dashes/now-with-dashes",
		"now.with.dots/simple",
		"now-with.dashes-and.dots/simple",
		"1-num.2-num/3-num",
		"1234/5678",
		"1.2.3.4/5678",
		"example.com/Uppercase_Is_OK_123",
		strings.Repeat("a", 253) + "/" + strings.Repeat("b", 63),
	}
	for i := range successCases {
		if errs := IsPrefixedLabelKey(successCases[i]); len(errs) != 0 {
			t.Errorf("case[%d]: %q: expected success: %v", i, successCases[i], errs)
		}
	}

	errorCases := []string{
		"simple",
		"now-with-dashes",
		"1-starts-with-num",
		"1234",
		"Uppercase_Is_OK_123",
		"requests.storage-foo",
		strings.Repeat("a", 63),
		"nospecialchars%^=@",
		"cantendwithadash-",
		"-cantstartwithadash-",
		"only/one/slash",
		"Example.com/abc",
		"example_com/abc",
		"example.com/",
		"/simple",
		strings.Repeat("a", 64),
		strings.Repeat("a", 254) + "/abc",
	}
	for i := range errorCases {
		if errs := IsPrefixedLabelKey(errorCases[i]); len(errs) == 0 {
			t.Errorf("case[%d]: %q: expected failure", i, errorCases[i])
		}
	}
}

func FuzzIsLabelKey(f *testing.F) {
	refKey := regexp.MustCompile("^" + labelKeyFmt + "$")
	for _, seed := range []string{"", "simple", "topology.kubernetes.io/zone", "/name", "prefix/", "a/b/c", "Example.com/name", strings.Repeat("a", 64), strings.Repeat("a", 254) + "/b", "a/b\u00e9"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, value string) {
		var want []string
		parts := strings.Split(value, "/")
		var name string
		switch len(parts) {
		case 1:
			name = parts[0]
		case 2:
			prefix := parts[0]
			name = parts[1]
			if len(prefix) == 0 {
				want = append(want, "prefix part "+EmptyError())
			} else if msgs := IsDNS1123Subdomain(prefix); len(msgs) != 0 {
				want = append(want, prefixEach(msgs, "prefix part ")...)
			}
		default:
			want = append(want, "a valid label key "+RegexError(labelKeyErrMsg, labelKeyFmt, "MyName", "my.name", "123-abc")+
				" with an optional DNS subdomain prefix and '/' (e.g. 'example.com/MyName')")
			if got := IsLabelKey(value); !slices.Equal(got, want) {
				t.Fatalf("IsLabelKey(%q) = %v, want %v", value, got, want)
			}
			return
		}
		if len(name) == 0 {
			want = append(want, "name part "+EmptyError())
		} else if len(name) > labelKeyMaxLength {
			want = append(want, "name part "+MaxLenError(labelKeyMaxLength))
		}
		if !refKey.MatchString(name) {
			want = append(want, "name part "+RegexError(labelKeyErrMsg, labelKeyFmt, "MyName", "my.name", "123-abc"))
		}
		if got := IsLabelKey(value); !slices.Equal(got, want) {
			t.Fatalf("IsLabelKey(%q) = %v, want %v", value, got, want)
		}
	})
}

func FuzzIsLabelValue(f *testing.F) {
	refVal := regexp.MustCompile("^" + labelValueFmt + "$")
	for _, seed := range []string{"", "us-central1-a", "My_Value.123", "-bad", "bad-", strings.Repeat("a", 63), strings.Repeat("a", 64), "a/b", "a\u00e9"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, value string) {
		var want []string
		if len(value) > LabelValueMaxLength {
			want = append(want, MaxLenError(LabelValueMaxLength))
		}
		if !refVal.MatchString(value) {
			want = append(want, RegexError(labelValueErrMsg, labelValueFmt, "MyValue", "my_value", "12345"))
		}
		if got := IsLabelValue(value); !slices.Equal(got, want) {
			t.Fatalf("IsLabelValue(%q) = %v, want %v", value, got, want)
		}
	})
}

func BenchmarkIsLabelKey(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsLabelKey("topology.kubernetes.io/zone")
	}
}

func BenchmarkIsLabelValue(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsLabelValue("us-central1-a")
	}
}
