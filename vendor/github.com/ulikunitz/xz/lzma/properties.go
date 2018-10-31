// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"errors"
	"fmt"
)

// maximum and minimum values for the LZMA properties.
const (
	minPB = 0
	maxPB = 4
)

// maxPropertyCode is the possible maximum of a properties code byte.
const maxPropertyCode = (maxPB+1)*(maxLP+1)*(maxLC+1) - 1

// Properties contains the parameters LC, LP and PB. The parameter LC
// defines the number of literal context bits; parameter LP the number
// of literal position bits and PB the number of position bits.
type Properties struct {
	LC int
	LP int
	PB int
}

// String returns the properties in a string representation.
func (p *Properties) String() string {
	return fmt.Sprintf("LC %d LP %d PB %d", p.LC, p.LP, p.PB)
}

// PropertiesForCode converts a properties code byte into a Properties value.
func PropertiesForCode(code byte) (p Properties, err error) {
	if code > maxPropertyCode {
		return p, errors.New("lzma: invalid properties code")
	}
	p.LC = int(code % 9)
	code /= 9
	p.LP = int(code % 5)
	code /= 5
	p.PB = int(code % 5)
	return p, err
}

// verify checks the properties for correctness.
func (p *Properties) verify() error {
	if p == nil {
		return errors.New("lzma: properties are nil")
	}
	if !(minLC <= p.LC && p.LC <= maxLC) {
		return errors.New("lzma: lc out of range")
	}
	if !(minLP <= p.LP && p.LP <= maxLP) {
		return errors.New("lzma: lp out of range")
	}
	if !(minPB <= p.PB && p.PB <= maxPB) {
		return errors.New("lzma: pb out of range")
	}
	return nil
}

// Code converts the properties to a byte. The function assumes that
// the properties components are all in range.
func (p Properties) Code() byte {
	return byte((p.PB*5+p.LP)*9 + p.LC)
}
