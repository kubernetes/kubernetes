package sdkuri

import "testing"

func TestPathJoin(t *testing.T) {
	cases := []struct {
		Elems  []string
		Expect string
	}{
		{Elems: []string{"/"}, Expect: "/"},
		{Elems: []string{}, Expect: ""},
		{Elems: []string{"blah", "el", "blah/"}, Expect: "blah/el/blah/"},
		{Elems: []string{"/asd", "asdfa", "asdfasd/"}, Expect: "/asd/asdfa/asdfasd/"},
		{Elems: []string{"asdfa", "asdfa", "asdfads"}, Expect: "asdfa/asdfa/asdfads"},
	}
	for _, c := range cases {
		if e, a := c.Expect, PathJoin(c.Elems...); e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
	}
}
