// +build go1.7

package ini

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidDataFiles(t *testing.T) {
	const expectedFileSuffix = "_expected"
	filepath.Walk("./testdata/valid", func(path string, info os.FileInfo, err error) error {
		if strings.HasSuffix(path, expectedFileSuffix) {
			return nil
		}

		if info.IsDir() {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			t.Errorf("%s: unexpected error, %v", path, err)
		}
		defer f.Close()

		tree, err := ParseAST(f)
		if err != nil {
			t.Errorf("%s: unexpected parse error, %v", path, err)
		}

		v := NewDefaultVisitor()
		err = Walk(tree, v)
		if err != nil {
			t.Errorf("%s: unexpected walk error, %v", path, err)
		}

		expectedPath := path + "_expected"
		e := map[string]interface{}{}

		b, err := ioutil.ReadFile(expectedPath)
		if err != nil {
			// ignore files that do not have an expected file
			return nil
		}

		err = json.Unmarshal(b, &e)
		if err != nil {
			t.Errorf("unexpected error during deserialization, %v", err)
		}

		for profile, tableIface := range e {
			p, ok := v.Sections.GetSection(profile)
			if !ok {
				t.Fatal("could not find profile " + profile)
			}

			table := tableIface.(map[string]interface{})
			for k, v := range table {
				switch e := v.(type) {
				case string:
					a := p.String(k)
					if e != a {
						t.Errorf("%s: expected %v, but received %v for profile %v", path, e, a, profile)
					}
				case int:
					a := p.Int(k)
					if int64(e) != a {
						t.Errorf("%s: expected %v, but received %v", path, e, a)
					}
				case float64:
					v := p.values[k]
					if v.Type == IntegerType {
						a := p.Int(k)
						if int64(e) != a {
							t.Errorf("%s: expected %v, but received %v", path, e, a)
						}
					} else {
						a := p.Float64(k)
						if e != a {
							t.Errorf("%s: expected %v, but received %v", path, e, a)
						}
					}
				default:
					t.Errorf("unexpected type: %T", e)
				}
			}
		}

		return nil
	})
}

func TestInvalidDataFiles(t *testing.T) {
	cases := []struct {
		path               string
		expectedParseError bool
		expectedWalkError  bool
	}{
		{
			path:               "./testdata/invalid/bad_syntax_1",
			expectedParseError: true,
		},
		{
			path:               "./testdata/invalid/incomplete_section_profile",
			expectedParseError: true,
		},
		{
			path:               "./testdata/invalid/syntax_error_comment",
			expectedParseError: true,
		},
	}

	for i, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			f, err := os.Open(c.path)
			if err != nil {
				t.Errorf("unexpected error, %v", err)
			}
			defer f.Close()

			tree, err := ParseAST(f)
			if err != nil && !c.expectedParseError {
				t.Errorf("%d: unexpected error, %v", i+1, err)
			} else if err == nil && c.expectedParseError {
				t.Errorf("%d: expected error, but received none", i+1)
			}

			if c.expectedParseError {
				return
			}

			v := NewDefaultVisitor()
			err = Walk(tree, v)
			if err == nil && c.expectedWalkError {
				t.Errorf("%d: expected error, but received none", i+1)
			}
		})
	}
}
