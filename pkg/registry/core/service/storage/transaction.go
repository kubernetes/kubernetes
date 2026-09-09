/*
Copyright 2020 The Kubernetes Authors.

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

package storage

// transaction represents something that may need to be finalized on success or
// failure of the larger transaction.
type transaction interface {
	// Commit tells the transaction to finalize any changes it may have
	// pending.  This cannot fail, so errors must be handled internally.
	Commit()

	// Revert tells the transaction to abandon or undo any changes it may have
	// pending.  This cannot fail, so errors must be handled internally.
	Revert()
}

// metaTransaction is a collection of transactions.
type metaTransaction []transaction

func (mt metaTransaction) Commit() {
	for _, t := range mt {
		t.Commit()
	}
}

func (mt metaTransaction) Revert() {
	for _, t := range mt {
		t.Revert()
	}
}

// callbackTransaction is a transaction which calls arbitrary functions.
type callbackTransaction struct {
	commit func()
	revert func()
}

func (cb callbackTransaction) Commit() {
	if cb.commit != nil {
		cb.commit()
	}
}

func (cb callbackTransaction) Revert() {
	if cb.revert != nil {
		cb.revert()
	}
}
