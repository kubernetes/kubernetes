/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	interpret(suffix) (base, exponent int, fmt Format, ok bool)
	construct(base, exponent int, fmt Format) (s suffix, ok bool)
}

// quantitySuffixer handles suffixes for all three formats that quantity
// can handle.
var quantitySuffixer = newSuffixer()

type bePair struct {
	base, exponent int
}

type listSuffixer struct {
	suffixToBE map[suffix]bePair
	beToSuffix map[bePair]suffix
}

func (ls *listSuffixer) addSuffix(s suffix, pair bePair) {
	if ls.suffixToBE == nil {
		ls.suffixToBE = map[suffix]bePair{}
	}
	if ls.beToSuffix == nil {
		ls.beToSuffix = map[bePair]suffix{}
	}
	ls.suffixToBE[s] = pair
	ls.beToSuffix[pair] = s
}

func (ls *listSuffixer) lookup(s suffix) (base, exponent int, ok bool) {
	pair, ok := ls.suffixToBE[s]
	if !ok {
		return 0, 0, false
	}
	return pair.base, pair.exponent, true
}

func (ls *listSuffixer) construct(base, exponent int) (s suffix, ok bool) {
	s, ok = ls.beToSuffix[bePair{base, exponent}]
	return
}

type suffixHandler struct {
	decSuffixes listSuffixer
	binSuffixes listSuffixer
}

func newSuffixer() suffixer {
	sh := &suffixHandler{}

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

	return sh
}

func (sh *suffixHandler) construct(base, exponent int, fmt Format) (s suffix, ok bool) {
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

func (sh *suffixHandler) interpret(suffix suffix) (base, exponent int, fmt Format, ok bool) {
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
		return 10, int(parsed), DecimalExponent, true
	}

	return 0, 0, DecimalExponent, false
}
