package filters

import (
	"sort"
	"testing"
)

func TestParseArgs(t *testing.T) {
	// equivalent of `docker ps -f 'created=today' -f 'image.name=ubuntu*' -f 'image.name=*untu'`
	flagArgs := []string{
		"created=today",
		"image.name=ubuntu*",
		"image.name=*untu",
	}
	var (
		args = Args{}
		err  error
	)
	for i := range flagArgs {
		args, err = ParseFlag(flagArgs[i], args)
		if err != nil {
			t.Errorf("failed to parse %s: %s", flagArgs[i], err)
		}
	}
	if len(args["created"]) != 1 {
		t.Errorf("failed to set this arg")
	}
	if len(args["image.name"]) != 2 {
		t.Errorf("the args should have collapsed")
	}
}

func TestParseArgsEdgeCase(t *testing.T) {
	var filters Args
	args, err := ParseFlag("", filters)
	if err != nil {
		t.Fatal(err)
	}
	if args == nil || len(args) != 0 {
		t.Fatalf("Expected an empty Args (map), got %v", args)
	}
	if args, err = ParseFlag("anything", args); err == nil || err != ErrorBadFormat {
		t.Fatalf("Expected ErrorBadFormat, got %v", err)
	}
}

func TestToParam(t *testing.T) {
	a := Args{
		"created":    []string{"today"},
		"image.name": []string{"ubuntu*", "*untu"},
	}

	_, err := ToParam(a)
	if err != nil {
		t.Errorf("failed to marshal the filters: %s", err)
	}
}

func TestFromParam(t *testing.T) {
	invalids := []string{
		"anything",
		"['a','list']",
		"{'key': 'value'}",
		`{"key": "value"}`,
	}
	valids := map[string]Args{
		`{"key": ["value"]}`: {
			"key": {"value"},
		},
		`{"key": ["value1", "value2"]}`: {
			"key": {"value1", "value2"},
		},
		`{"key1": ["value1"], "key2": ["value2"]}`: {
			"key1": {"value1"},
			"key2": {"value2"},
		},
	}
	for _, invalid := range invalids {
		if _, err := FromParam(invalid); err == nil {
			t.Fatalf("Expected an error with %v, got nothing", invalid)
		}
	}
	for json, expectedArgs := range valids {
		args, err := FromParam(json)
		if err != nil {
			t.Fatal(err)
		}
		if len(args) != len(expectedArgs) {
			t.Fatalf("Expected %v, go %v", expectedArgs, args)
		}
		for key, expectedValues := range expectedArgs {
			values := args[key]
			sort.Strings(values)
			sort.Strings(expectedValues)
			if len(values) != len(expectedValues) {
				t.Fatalf("Expected %v, go %v", expectedArgs, args)
			}
			for index, expectedValue := range expectedValues {
				if values[index] != expectedValue {
					t.Fatalf("Expected %v, go %v", expectedArgs, args)
				}
			}
		}
	}
}

func TestEmpty(t *testing.T) {
	a := Args{}
	v, err := ToParam(a)
	if err != nil {
		t.Errorf("failed to marshal the filters: %s", err)
	}
	v1, err := FromParam(v)
	if err != nil {
		t.Errorf("%s", err)
	}
	if len(a) != len(v1) {
		t.Errorf("these should both be empty sets")
	}
}

func TestArgsMatchKVList(t *testing.T) {
	// empty sources
	args := Args{
		"created": []string{"today"},
	}
	if args.MatchKVList("created", map[string]string{}) {
		t.Fatalf("Expected false for (%v,created), got true", args)
	}
	// Not empty sources
	sources := map[string]string{
		"key1": "value1",
		"key2": "value2",
		"key3": "value3",
	}
	matches := map[*Args]string{
		&Args{}: "field",
		&Args{
			"created": []string{"today"},
			"labels":  []string{"key1"},
		}: "labels",
		&Args{
			"created": []string{"today"},
			"labels":  []string{"key1=value1"},
		}: "labels",
	}
	differs := map[*Args]string{
		&Args{
			"created": []string{"today"},
		}: "created",
		&Args{
			"created": []string{"today"},
			"labels":  []string{"key4"},
		}: "labels",
		&Args{
			"created": []string{"today"},
			"labels":  []string{"key1=value3"},
		}: "labels",
	}
	for args, field := range matches {
		if args.MatchKVList(field, sources) != true {
			t.Fatalf("Expected true for %v on %v, got false", sources, args)
		}
	}
	for args, field := range differs {
		if args.MatchKVList(field, sources) != false {
			t.Fatalf("Expected false for %v on %v, got true", sources, args)
		}
	}
}

func TestArgsMatch(t *testing.T) {
	source := "today"
	matches := map[*Args]string{
		&Args{}: "field",
		&Args{
			"created": []string{"today"},
			"labels":  []string{"key1"},
		}: "today",
		&Args{
			"created": []string{"to*"},
		}: "created",
		&Args{
			"created": []string{"to(.*)"},
		}: "created",
		&Args{
			"created": []string{"tod"},
		}: "created",
		&Args{
			"created": []string{"anything", "to*"},
		}: "created",
	}
	differs := map[*Args]string{
		&Args{
			"created": []string{"tomorrow"},
		}: "created",
		&Args{
			"created": []string{"to(day"},
		}: "created",
		&Args{
			"created": []string{"tom(.*)"},
		}: "created",
		&Args{
			"created": []string{"today1"},
			"labels":  []string{"today"},
		}: "created",
	}
	for args, field := range matches {
		if args.Match(field, source) != true {
			t.Fatalf("Expected true for %v on %v, got false", source, args)
		}
	}
	for args, field := range differs {
		if args.Match(field, source) != false {
			t.Fatalf("Expected false for %v on %v, got true", source, args)
		}
	}
}
