package dockerfile

import (
	"strings"
	"testing"

	"github.com/docker/docker/builder/dockerfile/parser"
	"github.com/stretchr/testify/assert"
)

func TestAddNodesForLabelOption(t *testing.T) {
	dockerfile := "FROM scratch"
	result, err := parser.Parse(strings.NewReader(dockerfile))
	assert.NoError(t, err)

	labels := map[string]string{
		"org.e": "cli-e",
		"org.d": "cli-d",
		"org.c": "cli-c",
		"org.b": "cli-b",
		"org.a": "cli-a",
	}
	nodes := result.AST
	addNodesForLabelOption(nodes, labels)

	expected := []string{
		"FROM scratch",
		`LABEL "org.a"='cli-a' "org.b"='cli-b' "org.c"='cli-c' "org.d"='cli-d' "org.e"='cli-e'`,
	}
	assert.Len(t, nodes.Children, 2)
	for i, v := range nodes.Children {
		assert.Equal(t, expected[i], v.Original)
	}
}
