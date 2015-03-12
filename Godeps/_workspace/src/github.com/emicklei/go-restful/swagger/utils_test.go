package swagger

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func testJsonFromStruct(t *testing.T, sample interface{}, expectedJson string) {
	compareJson(t, false, modelsFromStruct(sample), expectedJson)
}

func modelsFromStruct(sample interface{}) map[string]Model {
	models := map[string]Model{}
	builder := modelBuilder{models}
	builder.addModel(reflect.TypeOf(sample), "")
	return models
}

func compareJson(t *testing.T, flatCompare bool, value interface{}, expectedJsonAsString string) {
	var output []byte
	var err error
	if flatCompare {
		output, err = json.Marshal(value)
	} else {
		output, err = json.MarshalIndent(value, " ", " ")
	}
	if err != nil {
		t.Error(err.Error())
		return
	}
	actual := string(output)
	var actualMap map[string]interface{}
	var expectedMap map[string]interface{}
	json.Unmarshal(output, &actualMap)
	json.Unmarshal([]byte(expectedJsonAsString), &expectedMap)
	if !reflect.DeepEqual(actualMap, expectedMap) {
		fmt.Println("---- expected -----")
		fmt.Println(withLineNumbers(expectedJsonAsString))
		fmt.Println("---- actual -----")
		fmt.Println(withLineNumbers(actual))
		fmt.Println("---- raw -----")
		fmt.Println(actual)
		t.Error("there are differences")
	}
}

func indexOfNonMatchingLine(actual, expected string) int {
	a := strings.Split(actual, "\n")
	e := strings.Split(expected, "\n")
	size := len(a)
	if len(e) < len(a) {
		size = len(e)
	}
	for i := 0; i < size; i++ {
		if a[i] != e[i] {
			return i
		}
	}
	return -1
}

func withLineNumbers(content string) string {
	var buffer bytes.Buffer
	lines := strings.Split(content, "\n")
	for i, each := range lines {
		buffer.WriteString(fmt.Sprintf("%d:%s\n", i, each))
	}
	return buffer.String()
}
