package filters

import (
	"errors"
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
		args = NewArgs()
		err  error
	)
	for i := range flagArgs {
		args, err = ParseFlag(flagArgs[i], args)
		if err != nil {
			t.Errorf("failed to parse %s: %s", flagArgs[i], err)
		}
	}
	if len(args.Get("created")) != 1 {
		t.Error("failed to set this arg")
	}
	if len(args.Get("image.name")) != 2 {
		t.Error("the args should have collapsed")
	}
}

func TestParseArgsEdgeCase(t *testing.T) {
	var filters Args
	args, err := ParseFlag("", filters)
	if err != nil {
		t.Fatal(err)
	}
	if args.Len() != 0 {
		t.Fatalf("Expected an empty Args (map), got %v", args)
	}
	if args, err = ParseFlag("anything", args); err == nil || err != ErrBadFormat {
		t.Fatalf("Expected ErrBadFormat, got %v", err)
	}
}

func TestToParam(t *testing.T) {
	fields := map[string]map[string]bool{
		"created":    {"today": true},
		"image.name": {"ubuntu*": true, "*untu": true},
	}
	a := Args{fields: fields}

	_, err := ToParam(a)
	if err != nil {
		t.Errorf("failed to marshal the filters: %s", err)
	}
}

func TestToParamWithVersion(t *testing.T) {
	fields := map[string]map[string]bool{
		"created":    {"today": true},
		"image.name": {"ubuntu*": true, "*untu": true},
	}
	a := Args{fields: fields}

	str1, err := ToParamWithVersion("1.21", a)
	if err != nil {
		t.Errorf("failed to marshal the filters with version < 1.22: %s", err)
	}
	str2, err := ToParamWithVersion("1.22", a)
	if err != nil {
		t.Errorf("failed to marshal the filters with version >= 1.22: %s", err)
	}
	if str1 != `{"created":["today"],"image.name":["*untu","ubuntu*"]}` &&
		str1 != `{"created":["today"],"image.name":["ubuntu*","*untu"]}` {
		t.Errorf("incorrectly marshaled the filters: %s", str1)
	}
	if str2 != `{"created":{"today":true},"image.name":{"*untu":true,"ubuntu*":true}}` &&
		str2 != `{"created":{"today":true},"image.name":{"ubuntu*":true,"*untu":true}}` {
		t.Errorf("incorrectly marshaled the filters: %s", str2)
	}
}

func TestFromParam(t *testing.T) {
	invalids := []string{
		"anything",
		"['a','list']",
		"{'key': 'value'}",
		`{"key": "value"}`,
	}
	valid := map[*Args][]string{
		{fields: map[string]map[string]bool{"key": {"value": true}}}: {
			`{"key": ["value"]}`,
			`{"key": {"value": true}}`,
		},
		{fields: map[string]map[string]bool{"key": {"value1": true, "value2": true}}}: {
			`{"key": ["value1", "value2"]}`,
			`{"key": {"value1": true, "value2": true}}`,
		},
		{fields: map[string]map[string]bool{"key1": {"value1": true}, "key2": {"value2": true}}}: {
			`{"key1": ["value1"], "key2": ["value2"]}`,
			`{"key1": {"value1": true}, "key2": {"value2": true}}`,
		},
	}

	for _, invalid := range invalids {
		if _, err := FromParam(invalid); err == nil {
			t.Fatalf("Expected an error with %v, got nothing", invalid)
		}
	}

	for expectedArgs, matchers := range valid {
		for _, json := range matchers {
			args, err := FromParam(json)
			if err != nil {
				t.Fatal(err)
			}
			if args.Len() != expectedArgs.Len() {
				t.Fatalf("Expected %v, go %v", expectedArgs, args)
			}
			for key, expectedValues := range expectedArgs.fields {
				values := args.Get(key)

				if len(values) != len(expectedValues) {
					t.Fatalf("Expected %v, go %v", expectedArgs, args)
				}

				for _, v := range values {
					if !expectedValues[v] {
						t.Fatalf("Expected %v, go %v", expectedArgs, args)
					}
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
	if a.Len() != v1.Len() {
		t.Error("these should both be empty sets")
	}
}

func TestArgsMatchKVListEmptySources(t *testing.T) {
	args := NewArgs()
	if !args.MatchKVList("created", map[string]string{}) {
		t.Fatalf("Expected true for (%v,created), got true", args)
	}

	args = Args{map[string]map[string]bool{"created": {"today": true}}}
	if args.MatchKVList("created", map[string]string{}) {
		t.Fatalf("Expected false for (%v,created), got true", args)
	}
}

func TestArgsMatchKVList(t *testing.T) {
	// Not empty sources
	sources := map[string]string{
		"key1": "value1",
		"key2": "value2",
		"key3": "value3",
	}

	matches := map[*Args]string{
		{}: "field",
		{map[string]map[string]bool{
			"created": {"today": true},
			"labels":  {"key1": true}},
		}: "labels",
		{map[string]map[string]bool{
			"created": {"today": true},
			"labels":  {"key1=value1": true}},
		}: "labels",
	}

	for args, field := range matches {
		if args.MatchKVList(field, sources) != true {
			t.Fatalf("Expected true for %v on %v, got false", sources, args)
		}
	}

	differs := map[*Args]string{
		{map[string]map[string]bool{
			"created": {"today": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"today": true},
			"labels":  {"key4": true}},
		}: "labels",
		{map[string]map[string]bool{
			"created": {"today": true},
			"labels":  {"key1=value3": true}},
		}: "labels",
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
		{}: "field",
		{map[string]map[string]bool{
			"created": {"today": true}},
		}: "today",
		{map[string]map[string]bool{
			"created": {"to*": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"to(.*)": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"tod": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"anything": true, "to*": true}},
		}: "created",
	}

	for args, field := range matches {
		if args.Match(field, source) != true {
			t.Fatalf("Expected true for %v on %v, got false", source, args)
		}
	}

	differs := map[*Args]string{
		{map[string]map[string]bool{
			"created": {"tomorrow": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"to(day": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"tom(.*)": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"tom": true}},
		}: "created",
		{map[string]map[string]bool{
			"created": {"today1": true},
			"labels":  {"today": true}},
		}: "created",
	}

	for args, field := range differs {
		if args.Match(field, source) != false {
			t.Fatalf("Expected false for %v on %v, got true", source, args)
		}
	}
}

func TestAdd(t *testing.T) {
	f := NewArgs()
	f.Add("status", "running")
	v := f.fields["status"]
	if len(v) != 1 || !v["running"] {
		t.Fatalf("Expected to include a running status, got %v", v)
	}

	f.Add("status", "paused")
	if len(v) != 2 || !v["paused"] {
		t.Fatalf("Expected to include a paused status, got %v", v)
	}
}

func TestDel(t *testing.T) {
	f := NewArgs()
	f.Add("status", "running")
	f.Del("status", "running")
	v := f.fields["status"]
	if v["running"] {
		t.Fatal("Expected to not include a running status filter, got true")
	}
}

func TestLen(t *testing.T) {
	f := NewArgs()
	if f.Len() != 0 {
		t.Fatal("Expected to not include any field")
	}
	f.Add("status", "running")
	if f.Len() != 1 {
		t.Fatal("Expected to include one field")
	}
}

func TestExactMatch(t *testing.T) {
	f := NewArgs()

	if !f.ExactMatch("status", "running") {
		t.Fatal("Expected to match `running` when there are no filters, got false")
	}

	f.Add("status", "running")
	f.Add("status", "pause*")

	if !f.ExactMatch("status", "running") {
		t.Fatal("Expected to match `running` with one of the filters, got false")
	}

	if f.ExactMatch("status", "paused") {
		t.Fatal("Expected to not match `paused` with one of the filters, got true")
	}
}

func TestOnlyOneExactMatch(t *testing.T) {
	f := NewArgs()

	if !f.UniqueExactMatch("status", "running") {
		t.Fatal("Expected to match `running` when there are no filters, got false")
	}

	f.Add("status", "running")

	if !f.UniqueExactMatch("status", "running") {
		t.Fatal("Expected to match `running` with one of the filters, got false")
	}

	if f.UniqueExactMatch("status", "paused") {
		t.Fatal("Expected to not match `paused` with one of the filters, got true")
	}

	f.Add("status", "pause")
	if f.UniqueExactMatch("status", "running") {
		t.Fatal("Expected to not match only `running` with two filters, got true")
	}
}

func TestInclude(t *testing.T) {
	f := NewArgs()
	if f.Include("status") {
		t.Fatal("Expected to not include a status key, got true")
	}
	f.Add("status", "running")
	if !f.Include("status") {
		t.Fatal("Expected to include a status key, got false")
	}
}

func TestValidate(t *testing.T) {
	f := NewArgs()
	f.Add("status", "running")

	valid := map[string]bool{
		"status":   true,
		"dangling": true,
	}

	if err := f.Validate(valid); err != nil {
		t.Fatal(err)
	}

	f.Add("bogus", "running")
	if err := f.Validate(valid); err == nil {
		t.Fatal("Expected to return an error, got nil")
	}
}

func TestWalkValues(t *testing.T) {
	f := NewArgs()
	f.Add("status", "running")
	f.Add("status", "paused")

	f.WalkValues("status", func(value string) error {
		if value != "running" && value != "paused" {
			t.Fatalf("Unexpected value %s", value)
		}
		return nil
	})

	err := f.WalkValues("status", func(value string) error {
		return errors.New("return")
	})
	if err == nil {
		t.Fatal("Expected to get an error, got nil")
	}

	err = f.WalkValues("foo", func(value string) error {
		return errors.New("return")
	})
	if err != nil {
		t.Fatalf("Expected to not iterate when the field doesn't exist, got %v", err)
	}
}

func TestFuzzyMatch(t *testing.T) {
	f := NewArgs()
	f.Add("container", "foo")

	cases := map[string]bool{
		"foo":    true,
		"foobar": true,
		"barfoo": false,
		"bar":    false,
	}
	for source, match := range cases {
		got := f.FuzzyMatch("container", source)
		if got != match {
			t.Fatalf("Expected %v, got %v: %s", match, got, source)
		}
	}
}
