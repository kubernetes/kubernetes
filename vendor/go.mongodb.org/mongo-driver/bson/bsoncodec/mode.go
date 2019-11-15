// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsoncodec

import "fmt"

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

// TransitionError is an error returned when an invalid progressing a
// ValueReader or ValueWriter state machine occurs.
type TransitionError struct {
	parent      mode
	current     mode
	destination mode
}

func (te TransitionError) Error() string {
	if te.destination == mode(0) {
		return fmt.Sprintf("invalid state transition: cannot read/write value while in %s", te.current)
	}
	if te.parent == mode(0) {
		return fmt.Sprintf("invalid state transition: %s -> %s", te.current, te.destination)
	}
	return fmt.Sprintf("invalid state transition: %s -> %s; parent %s", te.current, te.destination, te.parent)
}
