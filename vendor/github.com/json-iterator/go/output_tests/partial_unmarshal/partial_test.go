// NOTE: This test is different than most of the other JSON roundtrip tests.
package test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	fuzz "github.com/google/gofuzz"
	jsoniter "github.com/json-iterator/go"
)

func Test_PartialUnmarshal(t *testing.T) {
	fz := fuzz.New().MaxDepth(10).NilChance(0.3)
	for i := 0; i < 100; i++ {
		var before typeForTest1
		fz.Fuzz(&before)

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

		var afterStd typeForTest2
		err = json.Unmarshal(jbIter, &afterStd)
		if err != nil {
			t.Fatalf("failed to unmarshal with stdlib: %v\nvia:\n    %s",
				err, indent(jbIter, "    "))
		}
		var afterIter typeForTest2
		err = jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal(jbIter, &afterIter)
		if err != nil {
			t.Fatalf("failed to unmarshal with jsoniter: %v\nvia:\n    %s",
				err, indent(jbIter, "    "))
		}
		if fingerprint(afterStd) != fingerprint(afterIter) {
			t.Fatalf("unmarshal expected:\n    %s\ngot:\n    %s\nvia:\n    %s",
				dump(afterStd), dump(afterIter), indent(jbIter, "    "))
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
