/*
Copyright 2022 The Kubernetes Authors.

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

package flag

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMergeNamedFlagSetsNoArguments(t *testing.T) {
	assert.Len(t, MergeNamedFlagSets().FlagSets, 0)
}

func getNFS(n int) NamedFlagSets {
	nfs := NamedFlagSets{}
	fs := nfs.FlagSet(fmt.Sprintf("flagSet%d", n))
	fs.String(fmt.Sprintf("name%d", n), fmt.Sprintf("value%d", n), "")
	return nfs
}

func assertNFS(t *testing.T, nfs, merged NamedFlagSets) {
	require.Len(t, merged.FlagSets, 1)
	require.NotNil(t, merged.FlagSets["flagSet1"])
	assert.Equal(t, nfs.Order, merged.Order)
	s, err := merged.FlagSets["flagSet1"].GetString("name1")
	require.NoError(t, err)
	assert.Equal(t, "value1", s)
}

func TestMergeNamedFlagSetsOneArgument(t *testing.T) {
	nfs1 := getNFS(1)
	merged := MergeNamedFlagSets(nfs1)
	assertNFS(t, nfs1, merged)
}

func TestMergeNamedFlagSetsTwoArgumentsWithNameCollision(t *testing.T) {
	nfs1 := getNFS(1)
	nfs2 := getNFS(1)
	merged := MergeNamedFlagSets(nfs1, nfs2)
	assertNFS(t, nfs1, merged)
}

func TestMergeNamedFlagSetsTwoArguments(t *testing.T) {
	nfs1 := NamedFlagSets{}
	fs1 := nfs1.FlagSet("flagSet1")
	fs1.String("name1", "value1", "usage1")

	nfs2 := NamedFlagSets{}
	fs2 := nfs2.FlagSet("flagSet2")
	fs2.String("name2", "value2", "usage2")

	merged := MergeNamedFlagSets(nfs1, nfs2)

	require.Len(t, merged.FlagSets, 2)
	assert.Equal(t, append(nfs1.Order, nfs2.Order...), merged.Order)

	require.NotNil(t, fs1, merged.FlagSets["flagSet1"])
	s, err := merged.FlagSets["flagSet1"].GetString("name1")
	require.NoError(t, err)
	assert.Equal(t, "value1", s)

	require.NotNil(t, fs2, merged.FlagSets["flagSet2"])
	s, err = merged.FlagSets["flagSet2"].GetString("name2")
	require.NoError(t, err)
	assert.Equal(t, "value2", s)
}
