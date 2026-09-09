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

const decimalIntegerErrMsg string = "must be a valid decimal integer in canonical form"

// IsDecimalInteger validates that a string represents a decimal integer in strict canonical form.
// This means the string must be formatted exactly as a human would naturally write an integer,
// without any programming language conventions like leading zeros, plus signs, or alternate bases.
//
// valid values:"0" or Non-zero integers (i.e., "123", "-456") where the first digit is 1-9,
// followed by any digits 0-9.
//
// This validator is stricter than strconv.ParseInt, which accepts leading zeros values (i.e, "0700")
// and interprets them as decimal 700, potentially causing confusion with octal notation.
func IsDecimalInteger(value string) []string {
	n := len(value)
	if n == 0 {
		return []string{EmptyError()}
	}

	i := 0
	if value[0] == '-' {
		if n == 1 {
			return []string{decimalIntegerErrMsg}
		}
		i = 1
	}

	if value[i] == '0' {
		if n == 1 && i == 0 {
			return nil
		}
		return []string{decimalIntegerErrMsg}
	}

	if value[i] < '1' || value[i] > '9' {
		return []string{decimalIntegerErrMsg}
	}

	for i++; i < n; i++ {
		if value[i] < '0' || value[i] > '9' {
			return []string{decimalIntegerErrMsg}
		}
	}

	return nil
}
