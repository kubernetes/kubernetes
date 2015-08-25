package user

import (
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func TestUserParseLine(t *testing.T) {
	var (
		a, b string
		c    []string
		d    int
	)

	parseLine("", &a, &b)
	if a != "" || b != "" {
		t.Fatalf("a and b should be empty ('%v', '%v')", a, b)
	}

	parseLine("a", &a, &b)
	if a != "a" || b != "" {
		t.Fatalf("a should be 'a' and b should be empty ('%v', '%v')", a, b)
	}

	parseLine("bad boys:corny cows", &a, &b)
	if a != "bad boys" || b != "corny cows" {
		t.Fatalf("a should be 'bad boys' and b should be 'corny cows' ('%v', '%v')", a, b)
	}

	parseLine("", &c)
	if len(c) != 0 {
		t.Fatalf("c should be empty (%#v)", c)
	}

	parseLine("d,e,f:g:h:i,j,k", &c, &a, &b, &c)
	if a != "g" || b != "h" || len(c) != 3 || c[0] != "i" || c[1] != "j" || c[2] != "k" {
		t.Fatalf("a should be 'g', b should be 'h', and c should be ['i','j','k'] ('%v', '%v', '%#v')", a, b, c)
	}

	parseLine("::::::::::", &a, &b, &c)
	if a != "" || b != "" || len(c) != 0 {
		t.Fatalf("a, b, and c should all be empty ('%v', '%v', '%#v')", a, b, c)
	}

	parseLine("not a number", &d)
	if d != 0 {
		t.Fatalf("d should be 0 (%v)", d)
	}

	parseLine("b:12:c", &a, &d, &b)
	if a != "b" || b != "c" || d != 12 {
		t.Fatalf("a should be 'b' and b should be 'c', and d should be 12 ('%v', '%v', %v)", a, b, d)
	}
}

func TestUserParsePasswd(t *testing.T) {
	users, err := ParsePasswdFilter(strings.NewReader(`
root:x:0:0:root:/root:/bin/bash
adm:x:3:4:adm:/var/adm:/bin/false
this is just some garbage data
`), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(users) != 3 {
		t.Fatalf("Expected 3 users, got %v", len(users))
	}
	if users[0].Uid != 0 || users[0].Name != "root" {
		t.Fatalf("Expected users[0] to be 0 - root, got %v - %v", users[0].Uid, users[0].Name)
	}
	if users[1].Uid != 3 || users[1].Name != "adm" {
		t.Fatalf("Expected users[1] to be 3 - adm, got %v - %v", users[1].Uid, users[1].Name)
	}
}

func TestUserParseGroup(t *testing.T) {
	groups, err := ParseGroupFilter(strings.NewReader(`
root:x:0:root
adm:x:4:root,adm,daemon
this is just some garbage data
`), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(groups) != 3 {
		t.Fatalf("Expected 3 groups, got %v", len(groups))
	}
	if groups[0].Gid != 0 || groups[0].Name != "root" || len(groups[0].List) != 1 {
		t.Fatalf("Expected groups[0] to be 0 - root - 1 member, got %v - %v - %v", groups[0].Gid, groups[0].Name, len(groups[0].List))
	}
	if groups[1].Gid != 4 || groups[1].Name != "adm" || len(groups[1].List) != 3 {
		t.Fatalf("Expected groups[1] to be 4 - adm - 3 members, got %v - %v - %v", groups[1].Gid, groups[1].Name, len(groups[1].List))
	}
}

func TestValidGetExecUser(t *testing.T) {
	const passwdContent = `
root:x:0:0:root user:/root:/bin/bash
adm:x:42:43:adm:/var/adm:/bin/false
this is just some garbage data
`
	const groupContent = `
root:x:0:root
adm:x:43:
grp:x:1234:root,adm
this is just some garbage data
`
	defaultExecUser := ExecUser{
		Uid:   8888,
		Gid:   8888,
		Sgids: []int{8888},
		Home:  "/8888",
	}

	tests := []struct {
		ref      string
		expected ExecUser
	}{
		{
			ref: "root",
			expected: ExecUser{
				Uid:   0,
				Gid:   0,
				Sgids: []int{0, 1234},
				Home:  "/root",
			},
		},
		{
			ref: "adm",
			expected: ExecUser{
				Uid:   42,
				Gid:   43,
				Sgids: []int{1234},
				Home:  "/var/adm",
			},
		},
		{
			ref: "root:adm",
			expected: ExecUser{
				Uid:   0,
				Gid:   43,
				Sgids: defaultExecUser.Sgids,
				Home:  "/root",
			},
		},
		{
			ref: "adm:1234",
			expected: ExecUser{
				Uid:   42,
				Gid:   1234,
				Sgids: defaultExecUser.Sgids,
				Home:  "/var/adm",
			},
		},
		{
			ref: "42:1234",
			expected: ExecUser{
				Uid:   42,
				Gid:   1234,
				Sgids: defaultExecUser.Sgids,
				Home:  "/var/adm",
			},
		},
		{
			ref: "1337:1234",
			expected: ExecUser{
				Uid:   1337,
				Gid:   1234,
				Sgids: defaultExecUser.Sgids,
				Home:  defaultExecUser.Home,
			},
		},
		{
			ref: "1337",
			expected: ExecUser{
				Uid:   1337,
				Gid:   defaultExecUser.Gid,
				Sgids: defaultExecUser.Sgids,
				Home:  defaultExecUser.Home,
			},
		},
		{
			ref: "",
			expected: ExecUser{
				Uid:   defaultExecUser.Uid,
				Gid:   defaultExecUser.Gid,
				Sgids: defaultExecUser.Sgids,
				Home:  defaultExecUser.Home,
			},
		},
	}

	for _, test := range tests {
		passwd := strings.NewReader(passwdContent)
		group := strings.NewReader(groupContent)

		execUser, err := GetExecUser(test.ref, &defaultExecUser, passwd, group)
		if err != nil {
			t.Logf("got unexpected error when parsing '%s': %s", test.ref, err.Error())
			t.Fail()
			continue
		}

		if !reflect.DeepEqual(test.expected, *execUser) {
			t.Logf("got:      %#v", execUser)
			t.Logf("expected: %#v", test.expected)
			t.Fail()
			continue
		}
	}
}

func TestInvalidGetExecUser(t *testing.T) {
	const passwdContent = `
root:x:0:0:root user:/root:/bin/bash
adm:x:42:43:adm:/var/adm:/bin/false
this is just some garbage data
`
	const groupContent = `
root:x:0:root
adm:x:43:
grp:x:1234:root,adm
this is just some garbage data
`

	tests := []string{
		// No such user/group.
		"notuser",
		"notuser:notgroup",
		"root:notgroup",
		"notuser:adm",
		"8888:notgroup",
		"notuser:8888",

		// Invalid user/group values.
		"-1:0",
		"0:-3",
		"-5:-2",
	}

	for _, test := range tests {
		passwd := strings.NewReader(passwdContent)
		group := strings.NewReader(groupContent)

		execUser, err := GetExecUser(test, nil, passwd, group)
		if err == nil {
			t.Logf("got unexpected success when parsing '%s': %#v", test, execUser)
			t.Fail()
			continue
		}
	}
}

func TestGetExecUserNilSources(t *testing.T) {
	const passwdContent = `
root:x:0:0:root user:/root:/bin/bash
adm:x:42:43:adm:/var/adm:/bin/false
this is just some garbage data
`
	const groupContent = `
root:x:0:root
adm:x:43:
grp:x:1234:root,adm
this is just some garbage data
`

	defaultExecUser := ExecUser{
		Uid:   8888,
		Gid:   8888,
		Sgids: []int{8888},
		Home:  "/8888",
	}

	tests := []struct {
		ref           string
		passwd, group bool
		expected      ExecUser
	}{
		{
			ref:    "",
			passwd: false,
			group:  false,
			expected: ExecUser{
				Uid:   8888,
				Gid:   8888,
				Sgids: []int{8888},
				Home:  "/8888",
			},
		},
		{
			ref:    "root",
			passwd: true,
			group:  false,
			expected: ExecUser{
				Uid:   0,
				Gid:   0,
				Sgids: []int{8888},
				Home:  "/root",
			},
		},
		{
			ref:    "0",
			passwd: false,
			group:  false,
			expected: ExecUser{
				Uid:   0,
				Gid:   8888,
				Sgids: []int{8888},
				Home:  "/8888",
			},
		},
		{
			ref:    "0:0",
			passwd: false,
			group:  false,
			expected: ExecUser{
				Uid:   0,
				Gid:   0,
				Sgids: []int{8888},
				Home:  "/8888",
			},
		},
	}

	for _, test := range tests {
		var passwd, group io.Reader

		if test.passwd {
			passwd = strings.NewReader(passwdContent)
		}

		if test.group {
			group = strings.NewReader(groupContent)
		}

		execUser, err := GetExecUser(test.ref, &defaultExecUser, passwd, group)
		if err != nil {
			t.Logf("got unexpected error when parsing '%s': %s", test.ref, err.Error())
			t.Fail()
			continue
		}

		if !reflect.DeepEqual(test.expected, *execUser) {
			t.Logf("got:      %#v", execUser)
			t.Logf("expected: %#v", test.expected)
			t.Fail()
			continue
		}
	}
}

func TestGetAdditionalGroups(t *testing.T) {
	const groupContent = `
root:x:0:root
adm:x:43:
grp:x:1234:root,adm
adm:x:4343:root,adm-duplicate
this is just some garbage data
`
	tests := []struct {
		groups   []string
		expected []int
		hasError bool
	}{
		{
			// empty group
			groups:   []string{},
			expected: []int{},
		},
		{
			// single group
			groups:   []string{"adm"},
			expected: []int{43},
		},
		{
			// multiple groups
			groups:   []string{"adm", "grp"},
			expected: []int{43, 1234},
		},
		{
			// invalid group
			groups:   []string{"adm", "grp", "not-exist"},
			expected: nil,
			hasError: true,
		},
		{
			// group with numeric id
			groups:   []string{"43"},
			expected: []int{43},
		},
		{
			// group with unknown numeric id
			groups:   []string{"adm", "10001"},
			expected: []int{43, 10001},
		},
		{
			// groups specified twice with numeric and name
			groups:   []string{"adm", "43"},
			expected: []int{43},
		},
		{
			// groups with too small id
			groups:   []string{"-1"},
			expected: nil,
			hasError: true,
		},
		{
			// groups with too large id
			groups:   []string{strconv.Itoa(1 << 31)},
			expected: nil,
			hasError: true,
		},
	}

	for _, test := range tests {
		group := strings.NewReader(groupContent)

		gids, err := GetAdditionalGroups(test.groups, group)
		if test.hasError && err == nil {
			t.Errorf("Parse(%#v) expects error but has none", test)
			continue
		}
		if !test.hasError && err != nil {
			t.Errorf("Parse(%#v) has error %v", test, err)
			continue
		}
		sort.Sort(sort.IntSlice(gids))
		if !reflect.DeepEqual(gids, test.expected) {
			t.Errorf("Gids(%v), expect %v from groups %v", gids, test.expected, test.groups)
		}
	}
}
