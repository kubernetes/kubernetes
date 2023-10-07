/*
Copyright 2014 The Kubernetes Authors.

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

package resource

import (
	"strconv"
)

type suffix string

// suffixer can interpret and construct suffixes.
type suffixer interface {
	interpret(suffix) (base, exponent int32, fmt Format, ok bool)
	construct(base, exponent int32, fmt Format) (s suffix, ok bool)
	constructBytes(base, exponent int32, fmt Format) (s []byte, ok bool)
}

// quantitySuffixer handles suffixes for all three formats that quantity
// can handle.
var quantitySuffixer = newSuffixer()

type bePair struct {
	base, exponent int32
}

type listSuffixer struct {
	suffixToBE      map[suffix]bePair
	beToSuffix      map[bePair]suffix
	beToSuffixBytes map[bePair][]byte
}

func (ls *listSuffixer) addSuffix(s suffix, pair bePair) {
	if ls.suffixToBE == nil {
		ls.suffixToBE = map[suffix]bePair{}
	}
	if ls.beToSuffix == nil {
		ls.beToSuffix = map[bePair]suffix{}
	}
	if ls.beToSuffixBytes == nil {
		ls.beToSuffixBytes = map[bePair][]byte{}
	}
	ls.suffixToBE[s] = pair
	ls.beToSuffix[pair] = s
	ls.beToSuffixBytes[pair] = []byte(s)
}

func (ls *listSuffixer) lookup(s suffix) (base, exponent int32, ok bool) {
	pair, ok := ls.suffixToBE[s]
	if !ok {
		return 0, 0, false
	}
	return pair.base, pair.exponent, true
}

func (ls *listSuffixer) construct(base, exponent int32) (s suffix, ok bool) {
	s, ok = ls.beToSuffix[bePair{base, exponent}]
	return
}

func (ls *listSuffixer) constructBytes(base, exponent int32) (s []byte, ok bool) {
	s, ok = ls.beToSuffixBytes[bePair{base, exponent}]
	return
}

type suffixHandler struct {
	decSuffixes listSuffixer
	binSuffixes listSuffixer
}

type fastLookup struct {
	*suffixHandler
}

func (l fastLookup) interpret(s suffix) (base, exponent int32, format Format, ok bool) {
	switch s {
	case "":
		return 10, 0, DecimalSI, true
	case "n":
		return 10, -9, DecimalSI, true
	case "u":
		return 10, -6, DecimalSI, true
	case "m":
		return 10, -3, DecimalSI, true
	case "k":
		return 10, 3, DecimalSI, true
	case "M":
		return 10, 6, DecimalSI, true
	case "G":
		return 10, 9, DecimalSI, true
	}
	return l.suffixHandler.interpret(s)
}

func newSuffixer() suffixer {
	sh := &suffixHandler{}

	// IMPORTANT: if you change this section you must change fastLookup

	sh.binSuffixes.addSuffix("Ki", bePair{2, 10})
	sh.binSuffixes.addSuffix("Mi", bePair{2, 20})
	sh.binSuffixes.addSuffix("Gi", bePair{2, 30})
	sh.binSuffixes.addSuffix("Ti", bePair{2, 40})
	sh.binSuffixes.addSuffix("Pi", bePair{2, 50})
	sh.binSuffixes.addSuffix("Ei", bePair{2, 60})
	// Don't emit an error when trying to produce
	// a suffix for 2^0.
	sh.decSuffixes.addSuffix("", bePair{2, 0})

	sh.decSuffixes.addSuffix("n", bePair{10, -9})
	sh.decSuffixes.addSuffix("u", bePair{10, -6})
	sh.decSuffixes.addSuffix("m", bePair{10, -3})
	sh.decSuffixes.addSuffix("", bePair{10, 0})
	sh.decSuffixes.addSuffix("k", bePair{10, 3})
	sh.decSuffixes.addSuffix("M", bePair{10, 6})
	sh.decSuffixes.addSuffix("G", bePair{10, 9})
	sh.decSuffixes.addSuffix("T", bePair{10, 12})
	sh.decSuffixes.addSuffix("P", bePair{10, 15})
	sh.decSuffixes.addSuffix("E", bePair{10, 18})

	return fastLookup{sh}
}

func (sh *suffixHandler) construct(base, exponent int32, fmt Format) (s suffix, ok bool) {
	switch fmt {
	case DecimalSI:
		return sh.decSuffixes.construct(base, exponent)
	case BinarySI:
		return sh.binSuffixes.construct(base, exponent)
	case DecimalExponent:
		if base != 10 {
			return "", false
		}
		if exponent == 0 {
			return "", true
		}
		return suffix("e" + strconv.FormatInt(int64(exponent), 10)), true
	}
	return "", false
}

func (sh *suffixHandler) constructBytes(base, exponent int32, format Format) (s []byte, ok bool) {
	switch format {
	case DecimalSI:
		return sh.decSuffixes.constructBytes(base, exponent)
	case BinarySI:
		return sh.binSuffixes.constructBytes(base, exponent)
	case DecimalExponent:
		if base != 10 {
			return nil, false
		}
		if exponent == 0 {
			return nil, true
		}
		result := make([]byte, 8)
		result[0] = 'e'
		number := strconv.AppendInt(result[1:1], int64(exponent), 10)
		if &result[1] == &number[0] {
			return result[:1+len(number)], true
		}
		result = append(result[:1], number...)
		return result, true
	}
	return nil, false
}

func (sh *suffixHandler) interpret(suffix suffix) (base, exponent int32, fmt Format, ok bool) {
	// Try lookup tables first
	if b, e, ok := sh.decSuffixes.lookup(suffix); ok {
		return b, e, DecimalSI, true
	}
	if b, e, ok := sh.binSuffixes.lookup(suffix); ok {
		return b, e, BinarySI, true
	}

	if len(suffix) > 1 && (suffix[0] == 'E' || suffix[0] == 'e') {
		parsed, err := strconv.ParseInt(string(suffix[1:]), 10, 64)
		if err != nil {
			return 0, 0, DecimalExponent, false
		}
		return 10, int32(parsed), DecimalExponent, true
	}

	return 0, 0, DecimalExponent, false
}
