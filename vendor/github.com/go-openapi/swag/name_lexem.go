// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import "unicode"

type (
	nameLexem interface {
		GetUnsafeGoName() string
		GetOriginal() string
		IsInitialism() bool
	}

	initialismNameLexem struct {
		original          string
		matchedInitialism string
	}

	casualNameLexem struct {
		original string
	}
)

func newInitialismNameLexem(original, matchedInitialism string) *initialismNameLexem {
	return &initialismNameLexem{
		original:          original,
		matchedInitialism: matchedInitialism,
	}
}

func newCasualNameLexem(original string) *casualNameLexem {
	return &casualNameLexem{
		original: original,
	}
}

func (l *initialismNameLexem) GetUnsafeGoName() string {
	return l.matchedInitialism
}

func (l *casualNameLexem) GetUnsafeGoName() string {
	var first rune
	var rest string
	for i, orig := range l.original {
		if i == 0 {
			first = orig
			continue
		}
		if i > 0 {
			rest = l.original[i:]
			break
		}
	}
	if len(l.original) > 1 {
		return string(unicode.ToUpper(first)) + lower(rest)
	}

	return l.original
}

func (l *initialismNameLexem) GetOriginal() string {
	return l.original
}

func (l *casualNameLexem) GetOriginal() string {
	return l.original
}

func (l *initialismNameLexem) IsInitialism() bool {
	return true
}

func (l *casualNameLexem) IsInitialism() bool {
	return false
}
