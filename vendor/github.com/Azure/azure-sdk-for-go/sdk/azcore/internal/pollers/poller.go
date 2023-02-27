//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package pollers

// FinalStateVia is the enumerated type for the possible final-state-via values.
type FinalStateVia string

const (
	// FinalStateViaAzureAsyncOp indicates the final payload comes from the Azure-AsyncOperation URL.
	FinalStateViaAzureAsyncOp FinalStateVia = "azure-async-operation"

	// FinalStateViaLocation indicates the final payload comes from the Location URL.
	FinalStateViaLocation FinalStateVia = "location"

	// FinalStateViaOriginalURI indicates the final payload comes from the original URL.
	FinalStateViaOriginalURI FinalStateVia = "original-uri"

	// FinalStateViaOpLocation indicates the final payload comes from the Operation-Location URL.
	FinalStateViaOpLocation FinalStateVia = "operation-location"
)
