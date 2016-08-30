package consul

import (
	"reflect"
	"testing"

	"github.com/hashicorp/consul/acl"
	"github.com/hashicorp/consul/consul/structs"
)

func TestFilterDirEnt(t *testing.T) {
	policy, _ := acl.Parse(testFilterRules)
	aclR, _ := acl.New(acl.DenyAll(), policy)

	type tcase struct {
		in  []string
		out []string
	}
	cases := []tcase{
		tcase{
			in:  []string{"foo/test", "foo/priv/nope", "foo/other", "zoo"},
			out: []string{"foo/test", "foo/other"},
		},
		tcase{
			in:  []string{"abe", "lincoln"},
			out: nil,
		},
		tcase{
			in:  []string{"abe", "foo/1", "foo/2", "foo/3", "nope"},
			out: []string{"foo/1", "foo/2", "foo/3"},
		},
	}

	for _, tc := range cases {
		ents := structs.DirEntries{}
		for _, in := range tc.in {
			ents = append(ents, &structs.DirEntry{Key: in})
		}

		ents = FilterDirEnt(aclR, ents)
		var outL []string
		for _, e := range ents {
			outL = append(outL, e.Key)
		}

		if !reflect.DeepEqual(outL, tc.out) {
			t.Fatalf("bad: %#v %#v", outL, tc.out)
		}
	}
}

func TestKeys(t *testing.T) {
	policy, _ := acl.Parse(testFilterRules)
	aclR, _ := acl.New(acl.DenyAll(), policy)

	type tcase struct {
		in  []string
		out []string
	}
	cases := []tcase{
		tcase{
			in:  []string{"foo/test", "foo/priv/nope", "foo/other", "zoo"},
			out: []string{"foo/test", "foo/other"},
		},
		tcase{
			in:  []string{"abe", "lincoln"},
			out: []string{},
		},
		tcase{
			in:  []string{"abe", "foo/1", "foo/2", "foo/3", "nope"},
			out: []string{"foo/1", "foo/2", "foo/3"},
		},
	}

	for _, tc := range cases {
		out := FilterKeys(aclR, tc.in)
		if !reflect.DeepEqual(out, tc.out) {
			t.Fatalf("bad: %#v %#v", out, tc.out)
		}
	}
}

var testFilterRules = `
key "" {
	policy = "deny"
}
key "foo/" {
	policy = "read"
}
key "foo/priv/" {
	policy = "deny"
}
key "zip/" {
	policy = "read"
}
`
