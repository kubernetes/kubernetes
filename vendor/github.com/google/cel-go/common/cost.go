// Copyright 2022 Google LLC
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

package common

const (
	// SelectAndIdentCost is the cost of an operation that accesses an identifier or performs a select.
	SelectAndIdentCost = 1

	// ConstCost is the cost of an operation that accesses a constant.
	ConstCost = 0

	// ListCreateBaseCost is the base cost of any operation that creates a new list.
	ListCreateBaseCost = 10

	// MapCreateBaseCost is the base cost of any operation that creates a new map.
	MapCreateBaseCost = 30

	// StructCreateBaseCost is the base cost of any operation that creates a new struct.
	StructCreateBaseCost = 40

	// StringTraversalCostFactor is multiplied to a length of a string when computing the cost of traversing the entire
	// string once.
	StringTraversalCostFactor = 0.1

	// RegexStringLengthCostFactor is multiplied ot the length of a regex string pattern when computing the cost of
	// applying the regex to a string of unit cost.
	RegexStringLengthCostFactor = 0.25
)
