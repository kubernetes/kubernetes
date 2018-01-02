package image

import (
	"encoding/json"
	"sort"
	"strings"
	"testing"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/layer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const sampleImageJSON = `{
	"architecture": "amd64",
	"os": "linux",
	"config": {},
	"rootfs": {
		"type": "layers",
		"diff_ids": []
	}
}`

func TestNewFromJSON(t *testing.T) {
	img, err := NewFromJSON([]byte(sampleImageJSON))
	require.NoError(t, err)
	assert.Equal(t, sampleImageJSON, string(img.RawJSON()))
}

func TestNewFromJSONWithInvalidJSON(t *testing.T) {
	_, err := NewFromJSON([]byte("{}"))
	assert.EqualError(t, err, "invalid image JSON, no RootFS key")
}

func TestMarshalKeyOrder(t *testing.T) {
	b, err := json.Marshal(&Image{
		V1Image: V1Image{
			Comment:      "a",
			Author:       "b",
			Architecture: "c",
		},
	})
	assert.NoError(t, err)

	expectedOrder := []string{"architecture", "author", "comment"}
	var indexes []int
	for _, k := range expectedOrder {
		indexes = append(indexes, strings.Index(string(b), k))
	}

	if !sort.IntsAreSorted(indexes) {
		t.Fatal("invalid key order in JSON: ", string(b))
	}
}

func TestNewChildImageFromImageWithRootFS(t *testing.T) {
	rootFS := NewRootFS()
	rootFS.Append(layer.DiffID("ba5e"))
	parent := &Image{
		RootFS: rootFS,
		History: []History{
			NewHistory("a", "c", "r", false),
		},
	}
	childConfig := ChildConfig{
		DiffID:  layer.DiffID("abcdef"),
		Author:  "author",
		Comment: "comment",
		ContainerConfig: &container.Config{
			Cmd: []string{"echo", "foo"},
		},
		Config: &container.Config{},
	}

	newImage := NewChildImage(parent, childConfig, "platform")
	expectedDiffIDs := []layer.DiffID{layer.DiffID("ba5e"), layer.DiffID("abcdef")}
	assert.Equal(t, expectedDiffIDs, newImage.RootFS.DiffIDs)
	assert.Equal(t, childConfig.Author, newImage.Author)
	assert.Equal(t, childConfig.Config, newImage.Config)
	assert.Equal(t, *childConfig.ContainerConfig, newImage.ContainerConfig)
	assert.Equal(t, "platform", newImage.OS)
	assert.Equal(t, childConfig.Config, newImage.Config)

	assert.Len(t, newImage.History, 2)
	assert.Equal(t, childConfig.Comment, newImage.History[1].Comment)

	// RootFS should be copied not mutated
	assert.NotEqual(t, parent.RootFS.DiffIDs, newImage.RootFS.DiffIDs)
}
