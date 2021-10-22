// +build codegen

package api

import (
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

func TestResolvedReferences(t *testing.T) {
	json := `{
		"operations": {
			"OperationName": {
				"input": { "shape": "TestName" }
			}
		},
		"shapes": {
			"TestName": {
				"type": "structure",
				"members": {
					"memberName1": { "shape": "OtherTest" },
					"memberName2": { "shape": "OtherTest" }
				}
			},
			"OtherTest": { "type": "string" }
		}
	}`
	a := API{}
	a.AttachString(json)
	if len(a.Shapes["OtherTest"].refs) != 2 {
		t.Errorf("Expected %d, but received %d", 2, len(a.Shapes["OtherTest"].refs))
	}
}

func TestTrimModelServiceVersions(t *testing.T) {
	cases := []struct {
		Paths   []string
		Include []string
		Exclude []string
	}{
		{
			Paths: []string{
				filepath.Join("foo", "baz", "2018-01-02", "api-2.json"),
				filepath.Join("foo", "baz", "2019-01-02", "api-2.json"),
				filepath.Join("foo", "baz", "2017-01-02", "api-2.json"),
				filepath.Join("foo", "bar", "2019-01-02", "api-2.json"),
				filepath.Join("foo", "bar", "2013-04-02", "api-2.json"),
				filepath.Join("foo", "bar", "2019-01-03", "api-2.json"),
			},
			Include: []string{
				filepath.Join("foo", "baz", "2019-01-02", "api-2.json"),
				filepath.Join("foo", "bar", "2019-01-03", "api-2.json"),
			},
			Exclude: []string{
				filepath.Join("foo", "baz", "2018-01-02", "api-2.json"),
				filepath.Join("foo", "baz", "2017-01-02", "api-2.json"),
				filepath.Join("foo", "bar", "2019-01-02", "api-2.json"),
				filepath.Join("foo", "bar", "2013-04-02", "api-2.json"),
			},
		},
	}

	for i, c := range cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			include, exclude := TrimModelServiceVersions(c.Paths)
			if e, a := c.Include, include; !reflect.DeepEqual(e, a) {
				t.Errorf("expect include %v, got %v", e, a)
			}
			if e, a := c.Exclude, exclude; !reflect.DeepEqual(e, a) {
				t.Errorf("expect exclude %v, got %v", e, a)
			}
		})
	}
}
