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

func Test_Roundtrip(t *testing.T) {
	fz := fuzz.New().MaxDepth(10).NilChance(0.3)
	for i := 0; i < 100; i++ {
		var before typeForTest
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

		var afterStd typeForTest
		err = json.Unmarshal(jbIter, &afterStd)
		if err != nil {
			t.Fatalf("failed to unmarshal with stdlib: %v\nvia:\n    %s",
				err, indent(jbIter, "    "))
		}
		var afterIter typeForTest
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

func benchmarkMarshal(t *testing.B, name string, fn func(interface{}) ([]byte, error)) {
	t.ReportAllocs()
	t.ResetTimer()

	var obj typeForTest
	fz := fuzz.NewWithSeed(0).MaxDepth(10).NilChance(0.3)
	fz.Fuzz(&obj)
	for i := 0; i < t.N; i++ {
		jb, err := fn(obj)
		if err != nil {
			t.Fatalf("%s failed to marshal:\n input: %s\n  error: %v", name, dump(obj), err)
		}
		_ = jb
	}
}

func benchmarkUnmarshal(t *testing.B, name string, fn func(data []byte, v interface{}) error) {
	t.ReportAllocs()
	t.ResetTimer()

	var before typeForTest
	fz := fuzz.NewWithSeed(0).MaxDepth(10).NilChance(0.3)
	fz.Fuzz(&before)
	jb, err := json.Marshal(before)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	for i := 0; i < t.N; i++ {
		var after typeForTest
		err = fn(jb, &after)
		if err != nil {
			t.Fatalf("%s failed to unmarshal:\n  input: %q\n  error: %v", name, string(jb), err)
		}
	}
}

func BenchmarkStandardMarshal(t *testing.B) {
	benchmarkMarshal(t, "stdlib", json.Marshal)
}

func BenchmarkStandardUnmarshal(t *testing.B) {
	benchmarkUnmarshal(t, "stdlib", json.Unmarshal)
}

func BenchmarkJSONIterMarshalFastest(t *testing.B) {
	benchmarkMarshal(t, "jsoniter-fastest", jsoniter.ConfigFastest.Marshal)
}

func BenchmarkJSONIterUnmarshalFastest(t *testing.B) {
	benchmarkUnmarshal(t, "jsoniter-fastest", jsoniter.ConfigFastest.Unmarshal)
}

func BenchmarkJSONIterMarshalDefault(t *testing.B) {
	benchmarkMarshal(t, "jsoniter-default", jsoniter.Marshal)
}

func BenchmarkJSONIterUnmarshalDefault(t *testing.B) {
	benchmarkUnmarshal(t, "jsoniter-default", jsoniter.Unmarshal)
}

func BenchmarkJSONIterMarshalCompatible(t *testing.B) {
	benchmarkMarshal(t, "jsoniter-compat", jsoniter.ConfigCompatibleWithStandardLibrary.Marshal)
}

func BenchmarkJSONIterUnmarshalCompatible(t *testing.B) {
	benchmarkUnmarshal(t, "jsoniter-compat", jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal)
}
