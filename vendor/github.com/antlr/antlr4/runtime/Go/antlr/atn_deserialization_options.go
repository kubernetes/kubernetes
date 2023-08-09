// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import "errors"

var defaultATNDeserializationOptions = ATNDeserializationOptions{true, true, false}

type ATNDeserializationOptions struct {
	readOnly                      bool
	verifyATN                     bool
	generateRuleBypassTransitions bool
}

func (opts *ATNDeserializationOptions) ReadOnly() bool {
	return opts.readOnly
}

func (opts *ATNDeserializationOptions) SetReadOnly(readOnly bool) {
	if opts.readOnly {
		panic(errors.New("Cannot mutate read only ATNDeserializationOptions"))
	}
	opts.readOnly = readOnly
}

func (opts *ATNDeserializationOptions) VerifyATN() bool {
	return opts.verifyATN
}

func (opts *ATNDeserializationOptions) SetVerifyATN(verifyATN bool) {
	if opts.readOnly {
		panic(errors.New("Cannot mutate read only ATNDeserializationOptions"))
	}
	opts.verifyATN = verifyATN
}

func (opts *ATNDeserializationOptions) GenerateRuleBypassTransitions() bool {
	return opts.generateRuleBypassTransitions
}

func (opts *ATNDeserializationOptions) SetGenerateRuleBypassTransitions(generateRuleBypassTransitions bool) {
	if opts.readOnly {
		panic(errors.New("Cannot mutate read only ATNDeserializationOptions"))
	}
	opts.generateRuleBypassTransitions = generateRuleBypassTransitions
}

func DefaultATNDeserializationOptions() *ATNDeserializationOptions {
	return NewATNDeserializationOptions(&defaultATNDeserializationOptions)
}

func NewATNDeserializationOptions(other *ATNDeserializationOptions) *ATNDeserializationOptions {
	o := new(ATNDeserializationOptions)
	if other != nil {
		*o = *other
		o.readOnly = false
	}
	return o
}
