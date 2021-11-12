// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

var ATNDeserializationOptionsdefaultOptions = &ATNDeserializationOptions{true, false, false}

type ATNDeserializationOptions struct {
	readOnly                      bool
	verifyATN                     bool
	generateRuleBypassTransitions bool
}

func NewATNDeserializationOptions(CopyFrom *ATNDeserializationOptions) *ATNDeserializationOptions {
	o := new(ATNDeserializationOptions)

	if CopyFrom != nil {
		o.readOnly = CopyFrom.readOnly
		o.verifyATN = CopyFrom.verifyATN
		o.generateRuleBypassTransitions = CopyFrom.generateRuleBypassTransitions
	}

	return o
}
