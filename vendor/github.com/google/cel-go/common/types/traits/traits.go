// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package traits defines interfaces that a type may implement to participate
// in operator overloads and function dispatch.
package traits

const (
	// AdderType types provide a '+' operator overload.
	AdderType = 1 << iota

	// ComparerType types support ordering comparisons '<', '<=', '>', '>='.
	ComparerType

	// ContainerType types support 'in' operations.
	ContainerType

	// DividerType types support '/' operations.
	DividerType

	// FieldTesterType types support the detection of field value presence.
	FieldTesterType

	// IndexerType types support index access with dynamic values.
	IndexerType

	// IterableType types can be iterated over in comprehensions.
	IterableType

	// IteratorType types support iterator semantics.
	IteratorType

	// MatcherType types support pattern matching via 'matches' method.
	MatcherType

	// ModderType types support modulus operations '%'
	ModderType

	// MultiplierType types support '*' operations.
	MultiplierType

	// NegatorType types support either negation via '!' or '-'
	NegatorType

	// ReceiverType types support dynamic dispatch to instance methods.
	ReceiverType

	// SizerType types support the size() method.
	SizerType

	// SubtractorType type support '-' operations.
	SubtractorType
)
