package test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/davecgh/go-spew/spew"
	"github.com/google/gofuzz"
	"github.com/json-iterator/go"
	"reflect"
	"strings"
	"testing"
)

var testCases []interface{}
var asymmetricTestCases [][2]interface{}

type selectedSymmetricCase struct {
	testCase interface{}
}

func Test_symmetric(t *testing.T) {
	for _, testCase := range testCases {
		selectedSymmetricCase, found := testCase.(selectedSymmetricCase)
		if found {
			testCases = []interface{}{selectedSymmetricCase.testCase}
			break
		}
	}
	for _, testCase := range testCases {
		valType := reflect.TypeOf(testCase).Elem()
		t.Run(valType.String(), func(t *testing.T) {
			fz := fuzz.New().MaxDepth(10).NilChance(0.3)
			for i := 0; i < 100; i++ {
				beforePtrVal := reflect.New(valType)
				beforePtr := beforePtrVal.Interface()
				fz.Fuzz(beforePtr)
				before := beforePtrVal.Elem().Interface()

				jbStd, err := json.Marshal(before)
				if err != nil {
					t.Fatalf("failed to marshal with stdlib: %v", err)
				}
				if len(strings.TrimSpace(string(jbStd))) == 0 {
					t.Fatal("stdlib marshal produced empty result and no error")
				}
				jbIter, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(before)
				if err != nil {
					t.Fatalf("failed to marshal with jsoniter: %v", err)
				}
				if len(strings.TrimSpace(string(jbIter))) == 0 {
					t.Fatal("jsoniter marshal produced empty result and no error")
				}
				if string(jbStd) != string(jbIter) {
					t.Fatalf("marshal expected:\n    %s\ngot:\n    %s\nobj:\n    %s",
						indent(jbStd, "    "), indent(jbIter, "    "), dump(before))
				}

				afterStdPtrVal := reflect.New(valType)
				afterStdPtr := afterStdPtrVal.Interface()
				err = json.Unmarshal(jbIter, afterStdPtr)
				if err != nil {
					t.Fatalf("failed to unmarshal with stdlib: %v\nvia:\n    %s",
						err, indent(jbIter, "    "))
				}
				afterIterPtrVal := reflect.New(valType)
				afterIterPtr := afterIterPtrVal.Interface()
				err = jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal(jbIter, afterIterPtr)
				if err != nil {
					t.Fatalf("failed to unmarshal with jsoniter: %v\nvia:\n    %s",
						err, indent(jbIter, "    "))
				}
				afterStd := afterStdPtrVal.Elem().Interface()
				afterIter := afterIterPtrVal.Elem().Interface()
				if fingerprint(afterStd) != fingerprint(afterIter) {
					t.Fatalf("unmarshal expected:\n    %s\ngot:\n    %s\nvia:\n    %s",
						dump(afterStd), dump(afterIter), indent(jbIter, "    "))
				}
			}
		})
	}
}

func Test_asymmetric(t *testing.T) {
	for _, testCase := range asymmetricTestCases {
		fromType := reflect.TypeOf(testCase[0]).Elem()
		toType := reflect.TypeOf(testCase[1]).Elem()
		fz := fuzz.New().MaxDepth(10).NilChance(0.3)
		for i := 0; i < 100; i++ {
			beforePtrVal := reflect.New(fromType)
			beforePtr := beforePtrVal.Interface()
			fz.Fuzz(beforePtr)
			before := beforePtrVal.Elem().Interface()

			jbStd, err := json.Marshal(before)
			if err != nil {
				t.Fatalf("failed to marshal with stdlib: %v", err)
			}
			if len(strings.TrimSpace(string(jbStd))) == 0 {
				t.Fatal("stdlib marshal produced empty result and no error")
			}
			jbIter, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(before)
			if err != nil {
				t.Fatalf("failed to marshal with jsoniter: %v", err)
			}
			if len(strings.TrimSpace(string(jbIter))) == 0 {
				t.Fatal("jsoniter marshal produced empty result and no error")
			}
			if string(jbStd) != string(jbIter) {
				t.Fatalf("marshal expected:\n    %s\ngot:\n    %s\nobj:\n    %s",
					indent(jbStd, "    "), indent(jbIter, "    "), dump(before))
			}

			afterStdPtrVal := reflect.New(toType)
			afterStdPtr := afterStdPtrVal.Interface()
			err = json.Unmarshal(jbIter, afterStdPtr)
			if err != nil {
				t.Fatalf("failed to unmarshal with stdlib: %v\nvia:\n    %s",
					err, indent(jbIter, "    "))
			}
			afterIterPtrVal := reflect.New(toType)
			afterIterPtr := afterIterPtrVal.Interface()
			err = jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal(jbIter, afterIterPtr)
			if err != nil {
				t.Fatalf("failed to unmarshal with jsoniter: %v\nvia:\n    %s",
					err, indent(jbIter, "    "))
			}
			afterStd := afterStdPtrVal.Elem().Interface()
			afterIter := afterIterPtrVal.Elem().Interface()
			if fingerprint(afterStd) != fingerprint(afterIter) {
				t.Fatalf("unmarshal expected:\n    %s\ngot:\n    %s\nvia:\n    %s",
					dump(afterStd), dump(afterIter), indent(jbIter, "    "))
			}
		}
	}
}

const indentStr = ">  "

func fingerprint(obj interface{}) string {
	c := spew.ConfigState{
		SortKeys: true,
		SpewKeys: true,
	}
	return c.Sprintf("%v", obj)
}

func dump(obj interface{}) string {
	cfg := spew.ConfigState{
		Indent: indentStr,
	}
	return cfg.Sdump(obj)
}

func indent(src []byte, prefix string) string {
	var buf bytes.Buffer
	err := json.Indent(&buf, src, prefix, indentStr)
	if err != nil {
		return fmt.Sprintf("!!! %v", err)
	}
	return buf.String()
}
