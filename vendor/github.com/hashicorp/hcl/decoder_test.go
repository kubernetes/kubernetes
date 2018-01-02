package hcl

import (
	"io/ioutil"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/hashicorp/hcl/hcl/ast"
	"github.com/hashicorp/hcl/testhelper"
)

func TestDecode_interface(t *testing.T) {
	cases := []struct {
		File string
		Err  bool
		Out  interface{}
	}{
		{
			"basic.hcl",
			false,
			map[string]interface{}{
				"foo": "bar",
				"bar": "${file(\"bing/bong.txt\")}",
			},
		},
		{
			"basic_squish.hcl",
			false,
			map[string]interface{}{
				"foo":     "bar",
				"bar":     "${file(\"bing/bong.txt\")}",
				"foo-bar": "baz",
			},
		},
		{
			"empty.hcl",
			false,
			map[string]interface{}{
				"resource": []map[string]interface{}{
					map[string]interface{}{
						"foo": []map[string]interface{}{
							map[string]interface{}{},
						},
					},
				},
			},
		},
		{
			"tfvars.hcl",
			false,
			map[string]interface{}{
				"regularvar": "Should work",
				"map.key1":   "Value",
				"map.key2":   "Other value",
			},
		},
		{
			"escape.hcl",
			false,
			map[string]interface{}{
				"foo":          "bar\"baz\\n",
				"qux":          "back\\slash",
				"bar":          "new\nline",
				"qax":          `slash\:colon`,
				"nested":       `${HH\:mm\:ss}`,
				"nestedquotes": `${"\"stringwrappedinquotes\""}`,
			},
		},
		{
			"float.hcl",
			false,
			map[string]interface{}{
				"a": 1.02,
			},
		},
		{
			"multiline_bad.hcl",
			true,
			nil,
		},
		{
			"multiline_literal.hcl",
			false,
			map[string]interface{}{"multiline_literal": testhelper.Unix2dos(`hello
  world`)},
		},
		{
			"multiline_no_marker.hcl",
			true,
			nil,
		},
		{
			"multiline.hcl",
			false,
			map[string]interface{}{"foo": testhelper.Unix2dos("bar\nbaz\n")},
		},
		{
			"multiline_indented.hcl",
			false,
			map[string]interface{}{"foo": testhelper.Unix2dos("  bar\n  baz\n")},
		},
		{
			"multiline_no_hanging_indent.hcl",
			false,
			map[string]interface{}{"foo": testhelper.Unix2dos("  baz\n    bar\n      foo\n")},
		},
		{
			"multiline_no_eof.hcl",
			false,
			map[string]interface{}{"foo": testhelper.Unix2dos("bar\nbaz\n"), "key": "value"},
		},
		{
			"multiline.json",
			false,
			map[string]interface{}{"foo": "bar\nbaz"},
		},
		{
			"null_strings.json",
			false,
			map[string]interface{}{
				"module": []map[string]interface{}{
					map[string]interface{}{
						"app": []map[string]interface{}{
							map[string]interface{}{"foo": ""},
						},
					},
				},
			},
		},
		{
			"scientific.json",
			false,
			map[string]interface{}{
				"a": 1e-10,
				"b": 1e+10,
				"c": 1e10,
				"d": 1.2e-10,
				"e": 1.2e+10,
				"f": 1.2e10,
			},
		},
		{
			"scientific.hcl",
			false,
			map[string]interface{}{
				"a": 1e-10,
				"b": 1e+10,
				"c": 1e10,
				"d": 1.2e-10,
				"e": 1.2e+10,
				"f": 1.2e10,
			},
		},
		{
			"terraform_heroku.hcl",
			false,
			map[string]interface{}{
				"name": "terraform-test-app",
				"config_vars": []map[string]interface{}{
					map[string]interface{}{
						"FOO": "bar",
					},
				},
			},
		},
		{
			"structure_multi.hcl",
			false,
			map[string]interface{}{
				"foo": []map[string]interface{}{
					map[string]interface{}{
						"baz": []map[string]interface{}{
							map[string]interface{}{"key": 7},
						},
					},
					map[string]interface{}{
						"bar": []map[string]interface{}{
							map[string]interface{}{"key": 12},
						},
					},
				},
			},
		},
		{
			"structure_multi.json",
			false,
			map[string]interface{}{
				"foo": []map[string]interface{}{
					map[string]interface{}{
						"baz": []map[string]interface{}{
							map[string]interface{}{"key": 7},
						},
					},
					map[string]interface{}{
						"bar": []map[string]interface{}{
							map[string]interface{}{"key": 12},
						},
					},
				},
			},
		},
		{
			"list_of_maps.hcl",
			false,
			map[string]interface{}{
				"foo": []interface{}{
					map[string]interface{}{"somekey1": "someval1"},
					map[string]interface{}{"somekey2": "someval2", "someextrakey": "someextraval"},
				},
			},
		},
		{
			"assign_deep.hcl",
			false,
			map[string]interface{}{
				"resource": []interface{}{
					map[string]interface{}{
						"foo": []interface{}{
							map[string]interface{}{
								"bar": []map[string]interface{}{
									map[string]interface{}{}}}}}}},
		},
		{
			"structure_list.hcl",
			false,
			map[string]interface{}{
				"foo": []map[string]interface{}{
					map[string]interface{}{
						"key": 7,
					},
					map[string]interface{}{
						"key": 12,
					},
				},
			},
		},
		{
			"structure_list.json",
			false,
			map[string]interface{}{
				"foo": []map[string]interface{}{
					map[string]interface{}{
						"key": 7,
					},
					map[string]interface{}{
						"key": 12,
					},
				},
			},
		},
		{
			"structure_list_deep.json",
			false,
			map[string]interface{}{
				"bar": []map[string]interface{}{
					map[string]interface{}{
						"foo": []map[string]interface{}{
							map[string]interface{}{
								"name": "terraform_example",
								"ingress": []map[string]interface{}{
									map[string]interface{}{
										"from_port": 22,
									},
									map[string]interface{}{
										"from_port": 80,
									},
								},
							},
						},
					},
				},
			},
		},

		{
			"nested_block_comment.hcl",
			false,
			map[string]interface{}{
				"bar": "value",
			},
		},

		{
			"unterminated_block_comment.hcl",
			true,
			nil,
		},

		{
			"unterminated_brace.hcl",
			true,
			nil,
		},

		{
			"nested_provider_bad.hcl",
			true,
			nil,
		},

		{
			"object_list.json",
			false,
			map[string]interface{}{
				"resource": []map[string]interface{}{
					map[string]interface{}{
						"aws_instance": []map[string]interface{}{
							map[string]interface{}{
								"db": []map[string]interface{}{
									map[string]interface{}{
										"vpc": "foo",
										"provisioner": []map[string]interface{}{
											map[string]interface{}{
												"file": []map[string]interface{}{
													map[string]interface{}{
														"source":      "foo",
														"destination": "bar",
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Logf("Testing: %s", tc.File)
		d, err := ioutil.ReadFile(filepath.Join(fixtureDir, tc.File))
		if err != nil {
			t.Fatalf("err: %s", err)
		}

		var out interface{}
		err = Decode(&out, string(d))
		if (err != nil) != tc.Err {
			t.Fatalf("Input: %s\n\nError: %s", tc.File, err)
		}

		if !reflect.DeepEqual(out, tc.Out) {
			t.Fatalf("Input: %s. Actual, Expected.\n\n%#v\n\n%#v", tc.File, out, tc.Out)
		}

		var v interface{}
		err = Unmarshal(d, &v)
		if (err != nil) != tc.Err {
			t.Fatalf("Input: %s\n\nError: %s", tc.File, err)
		}

		if !reflect.DeepEqual(v, tc.Out) {
			t.Fatalf("Input: %s. Actual, Expected.\n\n%#v\n\n%#v", tc.File, out, tc.Out)
		}
	}
}

func TestDecode_interfaceInline(t *testing.T) {
	cases := []struct {
		Value string
		Err   bool
		Out   interface{}
	}{
		{"t t e{{}}", true, nil},
		{"t=0t d {}", true, map[string]interface{}{"t": 0}},
		{"v=0E0v d{}", true, map[string]interface{}{"v": float64(0)}},
	}

	for _, tc := range cases {
		t.Logf("Testing: %q", tc.Value)

		var out interface{}
		err := Decode(&out, tc.Value)
		if (err != nil) != tc.Err {
			t.Fatalf("Input: %q\n\nError: %s", tc.Value, err)
		}

		if !reflect.DeepEqual(out, tc.Out) {
			t.Fatalf("Input: %q. Actual, Expected.\n\n%#v\n\n%#v", tc.Value, out, tc.Out)
		}

		var v interface{}
		err = Unmarshal([]byte(tc.Value), &v)
		if (err != nil) != tc.Err {
			t.Fatalf("Input: %q\n\nError: %s", tc.Value, err)
		}

		if !reflect.DeepEqual(v, tc.Out) {
			t.Fatalf("Input: %q. Actual, Expected.\n\n%#v\n\n%#v", tc.Value, out, tc.Out)
		}
	}
}

func TestDecode_equal(t *testing.T) {
	cases := []struct {
		One, Two string
	}{
		{
			"basic.hcl",
			"basic.json",
		},
		{
			"float.hcl",
			"float.json",
		},
		/*
			{
				"structure.hcl",
				"structure.json",
			},
		*/
		{
			"structure.hcl",
			"structure_flat.json",
		},
		{
			"terraform_heroku.hcl",
			"terraform_heroku.json",
		},
	}

	for _, tc := range cases {
		p1 := filepath.Join(fixtureDir, tc.One)
		p2 := filepath.Join(fixtureDir, tc.Two)

		d1, err := ioutil.ReadFile(p1)
		if err != nil {
			t.Fatalf("err: %s", err)
		}

		d2, err := ioutil.ReadFile(p2)
		if err != nil {
			t.Fatalf("err: %s", err)
		}

		var i1, i2 interface{}
		err = Decode(&i1, string(d1))
		if err != nil {
			t.Fatalf("err: %s", err)
		}

		err = Decode(&i2, string(d2))
		if err != nil {
			t.Fatalf("err: %s", err)
		}

		if !reflect.DeepEqual(i1, i2) {
			t.Fatalf(
				"%s != %s\n\n%#v\n\n%#v",
				tc.One, tc.Two,
				i1, i2)
		}
	}
}

func TestDecode_flatMap(t *testing.T) {
	var val map[string]map[string]string

	err := Decode(&val, testReadFile(t, "structure_flatmap.hcl"))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := map[string]map[string]string{
		"foo": map[string]string{
			"foo": "bar",
			"key": "7",
		},
	}

	if !reflect.DeepEqual(val, expected) {
		t.Fatalf("Actual: %#v\n\nExpected: %#v", val, expected)
	}
}

func TestDecode_structure(t *testing.T) {
	type Embedded interface{}

	type V struct {
		Embedded `hcl:"-"`
		Key      int
		Foo      string
	}

	var actual V

	err := Decode(&actual, testReadFile(t, "flat.hcl"))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := V{
		Key: 7,
		Foo: "bar",
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("Actual: %#v\n\nExpected: %#v", actual, expected)
	}
}

func TestDecode_structurePtr(t *testing.T) {
	type V struct {
		Key int
		Foo string
	}

	var actual *V

	err := Decode(&actual, testReadFile(t, "flat.hcl"))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := &V{
		Key: 7,
		Foo: "bar",
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("Actual: %#v\n\nExpected: %#v", actual, expected)
	}
}

func TestDecode_structureArray(t *testing.T) {
	// This test is extracted from a failure in Consul (consul.io),
	// hence the interesting structure naming.

	type KeyPolicyType string

	type KeyPolicy struct {
		Prefix string `hcl:",key"`
		Policy KeyPolicyType
	}

	type Policy struct {
		Keys []KeyPolicy `hcl:"key,expand"`
	}

	expected := Policy{
		Keys: []KeyPolicy{
			KeyPolicy{
				Prefix: "",
				Policy: "read",
			},
			KeyPolicy{
				Prefix: "foo/",
				Policy: "write",
			},
			KeyPolicy{
				Prefix: "foo/bar/",
				Policy: "read",
			},
			KeyPolicy{
				Prefix: "foo/bar/baz",
				Policy: "deny",
			},
		},
	}

	files := []string{
		"decode_policy.hcl",
		"decode_policy.json",
	}

	for _, f := range files {
		var actual Policy

		err := Decode(&actual, testReadFile(t, f))
		if err != nil {
			t.Fatalf("Input: %s\n\nerr: %s", f, err)
		}

		if !reflect.DeepEqual(actual, expected) {
			t.Fatalf("Input: %s\n\nActual: %#v\n\nExpected: %#v", f, actual, expected)
		}
	}
}

func TestDecode_sliceExpand(t *testing.T) {
	type testInner struct {
		Name string `hcl:",key"`
		Key  string
	}

	type testStruct struct {
		Services []testInner `hcl:"service,expand"`
	}

	expected := testStruct{
		Services: []testInner{
			testInner{
				Name: "my-service-0",
				Key:  "value",
			},
			testInner{
				Name: "my-service-1",
				Key:  "value",
			},
		},
	}

	files := []string{
		"slice_expand.hcl",
	}

	for _, f := range files {
		t.Logf("Testing: %s", f)

		var actual testStruct
		err := Decode(&actual, testReadFile(t, f))
		if err != nil {
			t.Fatalf("Input: %s\n\nerr: %s", f, err)
		}

		if !reflect.DeepEqual(actual, expected) {
			t.Fatalf("Input: %s\n\nActual: %#v\n\nExpected: %#v", f, actual, expected)
		}
	}
}

func TestDecode_structureMap(t *testing.T) {
	// This test is extracted from a failure in Terraform (terraform.io),
	// hence the interesting structure naming.

	type hclVariable struct {
		Default     interface{}
		Description string
		Fields      []string `hcl:",decodedFields"`
	}

	type rawConfig struct {
		Variable map[string]hclVariable
	}

	expected := rawConfig{
		Variable: map[string]hclVariable{
			"foo": hclVariable{
				Default:     "bar",
				Description: "bar",
				Fields:      []string{"Default", "Description"},
			},

			"amis": hclVariable{
				Default: []map[string]interface{}{
					map[string]interface{}{
						"east": "foo",
					},
				},
				Fields: []string{"Default"},
			},
		},
	}

	files := []string{
		"decode_tf_variable.hcl",
		"decode_tf_variable.json",
	}

	for _, f := range files {
		t.Logf("Testing: %s", f)

		var actual rawConfig
		err := Decode(&actual, testReadFile(t, f))
		if err != nil {
			t.Fatalf("Input: %s\n\nerr: %s", f, err)
		}

		if !reflect.DeepEqual(actual, expected) {
			t.Fatalf("Input: %s\n\nActual: %#v\n\nExpected: %#v", f, actual, expected)
		}
	}
}

func TestDecode_interfaceNonPointer(t *testing.T) {
	var value interface{}
	err := Decode(value, testReadFile(t, "basic_int_string.hcl"))
	if err == nil {
		t.Fatal("should error")
	}
}

func TestDecode_intString(t *testing.T) {
	var value struct {
		Count int
	}

	err := Decode(&value, testReadFile(t, "basic_int_string.hcl"))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if value.Count != 3 {
		t.Fatalf("bad: %#v", value.Count)
	}
}

func TestDecode_Node(t *testing.T) {
	// given
	var value struct {
		Content ast.Node
		Nested  struct {
			Content ast.Node
		}
	}

	content := `
content {
	hello = "world"
}
`

	// when
	err := Decode(&value, content)

	// then
	if err != nil {
		t.Errorf("unable to decode content, %v", err)
		return
	}

	// verify ast.Node can be decoded later
	var v map[string]interface{}
	err = DecodeObject(&v, value.Content)
	if err != nil {
		t.Errorf("unable to decode content, %v", err)
		return
	}

	if v["hello"] != "world" {
		t.Errorf("expected mapping to be returned")
	}
}

func TestDecode_NestedNode(t *testing.T) {
	// given
	var value struct {
		Nested struct {
			Content ast.Node
		}
	}

	content := `
nested "content" {
	hello = "world"
}
`

	// when
	err := Decode(&value, content)

	// then
	if err != nil {
		t.Errorf("unable to decode content, %v", err)
		return
	}

	// verify ast.Node can be decoded later
	var v map[string]interface{}
	err = DecodeObject(&v, value.Nested.Content)
	if err != nil {
		t.Errorf("unable to decode content, %v", err)
		return
	}

	if v["hello"] != "world" {
		t.Errorf("expected mapping to be returned")
	}
}

// https://github.com/hashicorp/hcl/issues/60
func TestDecode_topLevelKeys(t *testing.T) {
	type Template struct {
		Source string
	}

	templates := struct {
		Templates []*Template `hcl:"template"`
	}{}

	err := Decode(&templates, `
	template {
	    source = "blah"
	}

	template {
	    source = "blahblah"
	}`)

	if err != nil {
		t.Fatal(err)
	}

	if templates.Templates[0].Source != "blah" {
		t.Errorf("bad source: %s", templates.Templates[0].Source)
	}

	if templates.Templates[1].Source != "blahblah" {
		t.Errorf("bad source: %s", templates.Templates[1].Source)
	}
}
