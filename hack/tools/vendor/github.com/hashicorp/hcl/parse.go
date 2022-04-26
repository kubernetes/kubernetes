package hcl

import (
	"fmt"

	"github.com/hashicorp/hcl/hcl/ast"
	hclParser "github.com/hashicorp/hcl/hcl/parser"
	jsonParser "github.com/hashicorp/hcl/json/parser"
)

// ParseBytes accepts as input byte slice and returns ast tree.
//
// Input can be either JSON or HCL
func ParseBytes(in []byte) (*ast.File, error) {
	return parse(in)
}

// ParseString accepts input as a string and returns ast tree.
func ParseString(input string) (*ast.File, error) {
	return parse([]byte(input))
}

func parse(in []byte) (*ast.File, error) {
	switch lexMode(in) {
	case lexModeHcl:
		return hclParser.Parse(in)
	case lexModeJson:
		return jsonParser.Parse(in)
	}

	return nil, fmt.Errorf("unknown config format")
}

// Parse parses the given input and returns the root object.
//
// The input format can be either HCL or JSON.
func Parse(input string) (*ast.File, error) {
	return parse([]byte(input))
}
