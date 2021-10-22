package sourceinfo

import (
	"testing"
)

//TestFindLineNumbers runs unit tests on the sourceinfo package
func TestFindLineNumbersV2(t *testing.T) {
	keys := []string{"paths", "/pets", "get", "parameters", "0", "name"}
	token := "limit"
	file := "../../examples/v2.0/yaml/petstore.yaml"
	result, err := FindNode(file, keys, token)
	if err != nil {
		t.Errorf("%+v\n", err)
	}
	if result.Line != 23 {
		t.Errorf("Given token \"limit\", FindYamlLine() returned %d, expected 23", result.Line)
	}

	keys = []string{"paths", "/pets/{petId}", "get", "parameters", "0", "name"}
	token = "petId"
	file = "../../examples/v2.0/yaml/petstore.yaml"
	result, err = FindNode(file, keys, token)
	if err != nil {
		t.Errorf("%+v\n", err)
	}
	if result.Line != 61 {
		t.Errorf("Given token \"petId\", FindYamlLine() returned %d, expected 61", result.Line)
	}
}

func TestFindLineNumbersV3(t *testing.T) {
	keys := []string{"paths", "/pets", "get", "parameters", "0", "name"}
	token := "limit"
	file := "../../examples/v3.0/yaml/petstore.yaml"
	result, err := FindNode(file, keys, token)
	if err != nil {
		t.Errorf("%+v\n", err)
	}
	if result.Line != 18 {
		t.Errorf("Given token \"limit\", FindYamlLine() returned %d, expected 18", result.Line)
	}

	keys = []string{"paths", "/pets/{petId}", "get", "parameters", "0", "name"}
	token = "petId"
	file = "../../examples/v3.0/yaml/petstore.yaml"
	result, err = FindNode(file, keys, token)
	if err != nil {
		t.Errorf("%+v\n", err)
	}
	if result.Line != 64 {
		t.Errorf("Given token \"petId\", FindYamlLine() returned %d, expected 64", result.Line)
	}
}
