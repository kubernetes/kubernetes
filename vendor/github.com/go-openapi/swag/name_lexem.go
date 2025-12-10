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

import (
	"unicode"
	"unicode/utf8"
)

type (
	lexemKind uint8

	nameLexem struct {
		original          string
		matchedInitialism string
		kind              lexemKind
	}
)

const (
	lexemKindCasualName lexemKind = iota
	lexemKindInitialismName
)

func newInitialismNameLexem(original, matchedInitialism string) nameLexem {
	return nameLexem{
		kind:              lexemKindInitialismName,
		original:          original,
		matchedInitialism: matchedInitialism,
	}
}

func newCasualNameLexem(original string) nameLexem {
	return nameLexem{
		kind:     lexemKindCasualName,
		original: original,
	}
}

func (l nameLexem) GetUnsafeGoName() string {
	if l.kind == lexemKindInitialismName {
		return l.matchedInitialism
	}

	var (
		first rune
		rest  string
	)

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
		b := poolOfBuffers.BorrowBuffer(utf8.UTFMax + len(rest))
		defer func() {
			poolOfBuffers.RedeemBuffer(b)
		}()
		b.WriteRune(unicode.ToUpper(first))
		b.WriteString(lower(rest))
		return b.String()
	}

	return l.original
}

func (l nameLexem) GetOriginal() string {
	return l.original
}

func (l nameLexem) IsInitialism() bool {
	return l.kind == lexemKindInitialismName
}
