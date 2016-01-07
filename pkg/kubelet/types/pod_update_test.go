/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package types

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGetValidatedSources(t *testing.T) {
	// Empty.
	sources, err := GetValidatedSources([]string{""})
	require.NoError(t, err)
	require.Len(t, sources, 0)

	// Success.
	sources, err = GetValidatedSources([]string{FileSource, ApiserverSource})
	require.NoError(t, err)
	require.Len(t, sources, 2)

	// All.
	sources, err = GetValidatedSources([]string{AllSource})
	require.NoError(t, err)
	require.Len(t, sources, 3)

	// Unknown source.
	sources, err = GetValidatedSources([]string{"taco"})
	require.Error(t, err)
}
