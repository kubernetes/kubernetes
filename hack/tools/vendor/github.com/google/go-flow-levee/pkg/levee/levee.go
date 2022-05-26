// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package levee exports the levee Analyzer.
package levee

import (
	"github.com/google/go-flow-levee/internal/pkg/config"
	"github.com/google/go-flow-levee/internal/pkg/levee"
	"github.com/google/go-flow-levee/internal/pkg/propagation/summary"
)

// Analyzer reports instances of source data reaching a sink.
var Analyzer = levee.Analyzer

// SetBytes is a wrapper around the config package's SetBytes function.
var SetBytes = config.SetBytes

// Summary is a wrapper around the propagation/summary
// package's Summary type.
type Summary = summary.Summary

// FuncSummaries is a wrapper around the propagation/summary
// package's map of regular function summaries.
var FuncSummaries = summary.FuncSummaries

// InterfaceFuncSummaries is a wrapper around the propagation/summary
// package's map of interface function summaries.
var InterfaceFuncSummaries = summary.InterfaceFuncSummaries
