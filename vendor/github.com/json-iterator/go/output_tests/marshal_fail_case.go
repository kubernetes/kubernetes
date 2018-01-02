package main

import (
	"encoding/json"
	"fmt"
	"reflect"

	jsoniter "github.com/json-iterator/go"
)

type typeForTest struct {
	F *float64
}

func main() {
	var obj typeForTest

	jb1, _ := json.Marshal(obj)
	jb2, _ := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(obj)
	if !reflect.DeepEqual(jb1, jb2) {
		fmt.Printf("results differ:\n  expected: `%s`\n       got: `%s`\n", string(jb1), string(jb2))
	}
}
