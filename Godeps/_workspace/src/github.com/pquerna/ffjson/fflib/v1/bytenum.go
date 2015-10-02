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

/* Portions of this file are on Go stdlib's strconv/iota.go */
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1

import (
	"github.com/pquerna/ffjson/fflib/v1/internal"
)

func ParseFloat(s []byte, bitSize int) (f float64, err error) {
	return internal.ParseFloat(s, bitSize)
}

// ParseUint is like ParseInt but for unsigned numbers, and oeprating on []byte
func ParseUint(s []byte, base int, bitSize int) (n uint64, err error) {
	if len(s) == 1 {
		switch s[0] {
		case '0':
			return 0, nil
		case '1':
			return 1, nil
		case '2':
			return 2, nil
		case '3':
			return 3, nil
		case '4':
			return 4, nil
		case '5':
			return 5, nil
		case '6':
			return 6, nil
		case '7':
			return 7, nil
		case '8':
			return 8, nil
		case '9':
			return 9, nil
		}
	}
	return internal.ParseUint(s, base, bitSize)
}

func ParseInt(s []byte, base int, bitSize int) (i int64, err error) {
	if len(s) == 1 {
		switch s[0] {
		case '0':
			return 0, nil
		case '1':
			return 1, nil
		case '2':
			return 2, nil
		case '3':
			return 3, nil
		case '4':
			return 4, nil
		case '5':
			return 5, nil
		case '6':
			return 6, nil
		case '7':
			return 7, nil
		case '8':
			return 8, nil
		case '9':
			return 9, nil
		}
	}
	return internal.ParseInt(s, base, bitSize)
}
