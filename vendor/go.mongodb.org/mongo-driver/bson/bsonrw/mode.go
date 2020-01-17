// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsonrw

import (
	"fmt"
)

type mode int

const (
	_ mode = iota
	mTopLevel
	mDocument
	mArray
	mValue
	mElement
	mCodeWithScope
	mSpacer
)

func (m mode) String() string {
	var str string

	switch m {
	case mTopLevel:
		str = "TopLevel"
	case mDocument:
		str = "DocumentMode"
	case mArray:
		str = "ArrayMode"
	case mValue:
		str = "ValueMode"
	case mElement:
		str = "ElementMode"
	case mCodeWithScope:
		str = "CodeWithScopeMode"
	case mSpacer:
		str = "CodeWithScopeSpacerFrame"
	default:
		str = "UnknownMode"
	}

	return str
}

func (m mode) TypeString() string {
	var str string

	switch m {
	case mTopLevel:
		str = "TopLevel"
	case mDocument:
		str = "Document"
	case mArray:
		str = "Array"
	case mValue:
		str = "Value"
	case mElement:
		str = "Element"
	case mCodeWithScope:
		str = "CodeWithScope"
	case mSpacer:
		str = "CodeWithScopeSpacer"
	default:
		str = "Unknown"
	}

	return str
}

// TransitionError is an error returned when an invalid progressing a
// ValueReader or ValueWriter state machine occurs.
// If read is false, the error is for writing
type TransitionError struct {
	name        string
	parent      mode
	current     mode
	destination mode
	modes       []mode
	action      string
}

func (te TransitionError) Error() string {
	errString := fmt.Sprintf("%s can only %s", te.name, te.action)
	if te.destination != mode(0) {
		errString = fmt.Sprintf("%s a %s", errString, te.destination.TypeString())
	}
	errString = fmt.Sprintf("%s while positioned on a", errString)
	for ind, m := range te.modes {
		if ind != 0 && len(te.modes) > 2 {
			errString = fmt.Sprintf("%s,", errString)
		}
		if ind == len(te.modes)-1 && len(te.modes) > 1 {
			errString = fmt.Sprintf("%s or", errString)
		}
		errString = fmt.Sprintf("%s %s", errString, m.TypeString())
	}
	errString = fmt.Sprintf("%s but is positioned on a %s", errString, te.current.TypeString())
	if te.parent != mode(0) {
		errString = fmt.Sprintf("%s with parent %s", errString, te.parent.TypeString())
	}
	return errString
}
