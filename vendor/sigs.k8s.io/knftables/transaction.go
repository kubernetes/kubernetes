/*
Copyright 2023 The Kubernetes Authors.

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

package knftables

import (
	"bytes"
	"fmt"
)

// Transaction represents an nftables transaction
type Transaction struct {
	*nftContext

	operations []operation
	err        error
}

// operation contains a single nftables operation (eg "add table", "flush chain")
type operation struct {
	verb verb
	obj  Object
}

// verb is used internally to represent the different "nft" verbs
type verb string

const (
	addVerb     verb = "add"
	createVerb  verb = "create"
	insertVerb  verb = "insert"
	replaceVerb verb = "replace"
	deleteVerb  verb = "delete"
	destroyVerb verb = "destroy"
	flushVerb   verb = "flush"
	resetVerb   verb = "reset"
)

// populateCommandBuf populates the transaction as series of nft commands to the given bytes.Buffer.
func (tx *Transaction) populateCommandBuf(buf *bytes.Buffer) {
	for _, op := range tx.operations {
		op.obj.writeOperation(op.verb, tx.nftContext, buf)
	}
}

// String returns the transaction as a string containing the nft commands; if there is
// a pending error, it will be output as a comment at the end of the transaction.
func (tx *Transaction) String() string {
	buf := &bytes.Buffer{}
	tx.populateCommandBuf(buf)
	if tx.err != nil {
		fmt.Fprintf(buf, "# ERROR: %v", tx.err)
	}

	return buf.String()
}

// NumOperations returns the number of operations queued in the transaction.
func (tx *Transaction) NumOperations() int {
	return len(tx.operations)
}

func (tx *Transaction) operation(verb verb, obj Object) {
	if tx.err != nil {
		return
	}
	if tx.err = obj.validate(verb, tx.nftContext); tx.err != nil {
		return
	}

	tx.operations = append(tx.operations, operation{verb: verb, obj: obj})
}

// Add adds an "nft add" operation to tx, ensuring that obj exists by creating it if it
// did not already exist. (If obj is a Rule, it will be appended to the end of its chain,
// or else added after the Rule indicated by this rule's Index or Handle.) The Add() call
// always succeeds, but if obj is invalid, or inconsistent with the existing nftables
// state, then an error will be returned when the transaction is Run.
func (tx *Transaction) Add(obj Object) {
	tx.operation(addVerb, obj)
}

// Create adds an "nft create" operation to tx, creating obj, which must not already
// exist. (If obj is a Rule, it will be appended to the end of its chain, or else added
// after the Rule indicated by this rule's Index or Handle.) The Create() call always
// succeeds, but if obj is invalid, already exists, or is inconsistent with the existing
// nftables state, then an error will be returned when the transaction is Run.
func (tx *Transaction) Create(obj Object) {
	tx.operation(createVerb, obj)
}

// Insert adds an "nft insert" operation to tx, inserting obj (which must be a Rule) at
// the start of its chain, or before the other Rule indicated by this rule's Index or
// Handle. The Insert() call always succeeds, but if obj is invalid or is inconsistent
// with the existing nftables state, then an error will be returned when the transaction
// is Run.
func (tx *Transaction) Insert(obj Object) {
	tx.operation(insertVerb, obj)
}

// Replace adds an "nft replace" operation to tx, replacing an existing rule with obj
// (which must be a Rule). The Replace() call always succeeds, but if obj is invalid, does
// not contain the Handle of an existing rule, or is inconsistent with the existing
// nftables state, then an error will be returned when the transaction is Run.
func (tx *Transaction) Replace(obj Object) {
	tx.operation(replaceVerb, obj)
}

// Flush adds an "nft flush" operation to tx, clearing the contents of obj. The Flush()
// call always succeeds, but if obj does not exist (or does not support flushing) then an
// error will be returned when the transaction is Run.
func (tx *Transaction) Flush(obj Object) {
	tx.operation(flushVerb, obj)
}

// Delete adds an "nft delete" operation to tx, deleting obj, which must exist. The
// Delete() call always succeeds, but if obj does not exist or cannot be deleted based on
// the information provided (eg, Handle is required but not set) then an error will be
// returned when the transaction is Run.
func (tx *Transaction) Delete(obj Object) {
	tx.operation(deleteVerb, obj)
}

// Reset adds a "nft reset" operation to tx, resetting obj (which must be a Counter).
// The Reset() call always succeeds, but if obj does not exist then an error will be
// returned when the transaction is Run.
func (tx *Transaction) Reset(obj Object) {
	tx.operation(resetVerb, obj)
}

// Destroy adds an "nft destroy" operation to tx, ensuring that obj does not exist, by
// deleting it if it does exist. The Destroy() call always succeeds, but if obj cannot be
// deleted based on the information provided (eg, Handle is required but not set) then an
// error will be returned when the transaction is Run.
//
// Support for the actual "nft destroy" command requires kernel 6.3+ and nft 1.0.7+. You
// can create the Interface with the `RequireDestroy` option if you want construction to
// fail on older hosts. Alternatively, you can create the interface with the
// `EmulateDestroy` option, in which case knftables will emulate Destroy by doing an
// Add+Delete. In that case, obj must be valid for both an Add and a Delete. (Even if the
// system you are on supports destroy, you may only call Destroy() in a
// backward-compatible way if you are using `EmulateDestroy`.) In particular, this means:
//
//   - You can only Destroy() objects by Name or Key, not by Handle.
//   - You can't Destroy() a Rule (since they can only be deleted by Handle).
//   - You do not need to include optional values in obj (e.g. base chain properties) but
//     if you do include them, they need to be correct.
//   - When Destroy()ing a Set or Map you must include the correct Type.
//   - When Destroy()ing a Map Element you must include the correct Value.
func (tx *Transaction) Destroy(obj Object) {
	if tx.err != nil {
		return
	}
	if tx.err = obj.validate(destroyVerb, tx.nftContext); tx.err != nil {
		return
	}

	if tx.emulateDestroy {
		err := obj.validate(addVerb, tx.nftContext)
		if err == nil {
			err = obj.validate(deleteVerb, tx.nftContext)
		}
		if err != nil {
			tx.err = fmt.Errorf("object is not compatible with EmulateDestroy: %w", err)
			return
		}
	}

	if tx.emulateDestroy && !tx.nftContext.hasDestroy {
		tx.operations = append(tx.operations, operation{verb: addVerb, obj: obj})
		tx.operations = append(tx.operations, operation{verb: deleteVerb, obj: obj})
	} else {
		tx.operations = append(tx.operations, operation{verb: destroyVerb, obj: obj})
	}
}
