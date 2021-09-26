package main

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/golang/mock/mockgen/model"
)

func TestMakeArgString(t *testing.T) {
	testCases := []struct {
		argNames  []string
		argTypes  []string
		argString string
	}{
		{
			argNames:  nil,
			argTypes:  nil,
			argString: "",
		},
		{
			argNames:  []string{"arg0"},
			argTypes:  []string{"int"},
			argString: "arg0 int",
		},
		{
			argNames:  []string{"arg0", "arg1"},
			argTypes:  []string{"int", "bool"},
			argString: "arg0 int, arg1 bool",
		},
		{
			argNames:  []string{"arg0", "arg1"},
			argTypes:  []string{"int", "int"},
			argString: "arg0, arg1 int",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2"},
			argTypes:  []string{"bool", "int", "int"},
			argString: "arg0 bool, arg1, arg2 int",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2"},
			argTypes:  []string{"int", "bool", "int"},
			argString: "arg0 int, arg1 bool, arg2 int",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2"},
			argTypes:  []string{"int", "int", "bool"},
			argString: "arg0, arg1 int, arg2 bool",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2"},
			argTypes:  []string{"int", "int", "int"},
			argString: "arg0, arg1, arg2 int",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3"},
			argTypes:  []string{"bool", "int", "int", "int"},
			argString: "arg0 bool, arg1, arg2, arg3 int",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3"},
			argTypes:  []string{"int", "bool", "int", "int"},
			argString: "arg0 int, arg1 bool, arg2, arg3 int",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3"},
			argTypes:  []string{"int", "int", "bool", "int"},
			argString: "arg0, arg1 int, arg2 bool, arg3 int",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3"},
			argTypes:  []string{"int", "int", "int", "bool"},
			argString: "arg0, arg1, arg2 int, arg3 bool",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3", "arg4"},
			argTypes:  []string{"bool", "int", "int", "int", "bool"},
			argString: "arg0 bool, arg1, arg2, arg3 int, arg4 bool",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3", "arg4"},
			argTypes:  []string{"int", "bool", "int", "int", "bool"},
			argString: "arg0 int, arg1 bool, arg2, arg3 int, arg4 bool",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3", "arg4"},
			argTypes:  []string{"int", "int", "bool", "int", "bool"},
			argString: "arg0, arg1 int, arg2 bool, arg3 int, arg4 bool",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3", "arg4"},
			argTypes:  []string{"int", "int", "int", "bool", "bool"},
			argString: "arg0, arg1, arg2 int, arg3, arg4 bool",
		},
		{
			argNames:  []string{"arg0", "arg1", "arg2", "arg3", "arg4"},
			argTypes:  []string{"int", "int", "bool", "bool", "int"},
			argString: "arg0, arg1 int, arg2, arg3 bool, arg4 int",
		},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			s := makeArgString(tc.argNames, tc.argTypes)
			if s != tc.argString {
				t.Errorf("result == %q, want %q", s, tc.argString)
			}
		})
	}
}

func TestNewIdentifierAllocator(t *testing.T) {
	a := newIdentifierAllocator([]string{"taken1", "taken2"})
	if len(a) != 2 {
		t.Fatalf("expected 2 items, got %v", len(a))
	}

	_, ok := a["taken1"]
	if !ok {
		t.Errorf("allocator doesn't contain 'taken1': %#v", a)
	}

	_, ok = a["taken2"]
	if !ok {
		t.Errorf("allocator doesn't contain 'taken2': %#v", a)
	}
}

func allocatorContainsIdentifiers(a identifierAllocator, ids []string) bool {
	if len(a) != len(ids) {
		return false
	}

	for _, id := range ids {
		_, ok := a[id]
		if !ok {
			return false
		}
	}

	return true
}

func TestIdentifierAllocator_allocateIdentifier(t *testing.T) {
	a := newIdentifierAllocator([]string{"taken"})

	t2 := a.allocateIdentifier("taken_2")
	if t2 != "taken_2" {
		t.Fatalf("expected 'taken_2', got %q", t2)
	}
	expected := []string{"taken", "taken_2"}
	if !allocatorContainsIdentifiers(a, expected) {
		t.Fatalf("allocator doesn't contain the expected items - allocator: %#v, expected items: %#v", a, expected)
	}

	t3 := a.allocateIdentifier("taken")
	if t3 != "taken_3" {
		t.Fatalf("expected 'taken_3', got %q", t3)
	}
	expected = []string{"taken", "taken_2", "taken_3"}
	if !allocatorContainsIdentifiers(a, expected) {
		t.Fatalf("allocator doesn't contain the expected items - allocator: %#v, expected items: %#v", a, expected)
	}

	t4 := a.allocateIdentifier("taken")
	if t4 != "taken_4" {
		t.Fatalf("expected 'taken_4', got %q", t4)
	}
	expected = []string{"taken", "taken_2", "taken_3", "taken_4"}
	if !allocatorContainsIdentifiers(a, expected) {
		t.Fatalf("allocator doesn't contain the expected items - allocator: %#v, expected items: %#v", a, expected)
	}

	id := a.allocateIdentifier("id")
	if id != "id" {
		t.Fatalf("expected 'id', got %q", id)
	}
	expected = []string{"taken", "taken_2", "taken_3", "taken_4", "id"}
	if !allocatorContainsIdentifiers(a, expected) {
		t.Fatalf("allocator doesn't contain the expected items - allocator: %#v, expected items: %#v", a, expected)
	}
}

func TestGenerateMockInterface_Helper(t *testing.T) {
	for _, test := range []struct {
		Name       string
		Identifier string
		HelperLine string
		Methods    []*model.Method
	}{
		{Name: "mock", Identifier: "MockSomename", HelperLine: "m.ctrl.T.Helper()"},
		{Name: "recorder", Identifier: "MockSomenameMockRecorder", HelperLine: "mr.mock.ctrl.T.Helper()"},
		{
			Name:       "mock identifier conflict",
			Identifier: "MockSomename",
			HelperLine: "m_2.ctrl.T.Helper()",
			Methods: []*model.Method{
				{
					Name: "MethodA",
					In: []*model.Parameter{
						{
							Name: "m",
							Type: &model.NamedType{Type: "int"},
						},
					},
				},
			},
		},
		{
			Name:       "recorder identifier conflict",
			Identifier: "MockSomenameMockRecorder",
			HelperLine: "mr_2.mock.ctrl.T.Helper()",
			Methods: []*model.Method{
				{
					Name: "MethodA",
					In: []*model.Parameter{
						{
							Name: "mr",
							Type: &model.NamedType{Type: "int"},
						},
					},
				},
			},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			g := generator{}

			if len(test.Methods) == 0 {
				test.Methods = []*model.Method{
					{Name: "MethodA"},
					{Name: "MethodB"},
				}
			}

			if err := g.GenerateMockInterface(&model.Interface{
				Name:    "Somename",
				Methods: test.Methods,
			}, "somepackage"); err != nil {
				t.Fatal(err)
			}

			lines := strings.Split(g.buf.String(), "\n")

			// T.Helper() should be the first line
			for _, method := range test.Methods {
				if strings.TrimSpace(lines[findMethod(t, test.Identifier, method.Name, lines)+1]) != test.HelperLine {
					t.Fatalf("method %s.%s did not declare itself a Helper method", test.Identifier, method.Name)
				}
			}
		})
	}
}

func findMethod(t *testing.T, identifier, methodName string, lines []string) int {
	t.Helper()
	r := regexp.MustCompile(fmt.Sprintf(`func\s+\(.+%s\)\s*%s`, identifier, methodName))
	for i, line := range lines {
		if r.MatchString(line) {
			return i
		}
	}

	t.Fatalf("unable to find 'func (m %s) %s'", identifier, methodName)
	panic("unreachable")
}

func TestGetArgNames(t *testing.T) {
	for _, testCase := range []struct {
		name     string
		method   *model.Method
		expected []string
	}{
		{
			name: "NamedArg",
			method: &model.Method{
				In: []*model.Parameter{
					{
						Name: "firstArg",
						Type: &model.NamedType{Type: "int"},
					},
					{
						Name: "secondArg",
						Type: &model.NamedType{Type: "string"},
					},
				},
			},
			expected: []string{"firstArg", "secondArg"},
		},
		{
			name: "NotNamedArg",
			method: &model.Method{
				In: []*model.Parameter{
					{
						Name: "",
						Type: &model.NamedType{Type: "int"},
					},
					{
						Name: "",
						Type: &model.NamedType{Type: "string"},
					},
				},
			},
			expected: []string{"arg0", "arg1"},
		},
		{
			name: "MixedNameArg",
			method: &model.Method{
				In: []*model.Parameter{
					{
						Name: "firstArg",
						Type: &model.NamedType{Type: "int"},
					},
					{
						Name: "_",
						Type: &model.NamedType{Type: "string"},
					},
				},
			},
			expected: []string{"firstArg", "arg1"},
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			g := generator{}

			result := g.getArgNames(testCase.method)
			if !reflect.DeepEqual(result, testCase.expected) {
				t.Fatalf("expected %s, got %s", result, testCase.expected)
			}
		})
	}
}

func Test_createPackageMap(t *testing.T) {
	tests := []struct {
		name            string
		importPath      string
		wantPackageName string
		wantOK          bool
	}{
		{"golang package", "context", "context", true},
		{"third party", "golang.org/x/tools/present", "present", true},
		{"modules", "rsc.io/quote/v3", "quote", true},
		{"fail", "this/should/not/work", "", false},
	}
	var importPaths []string
	for _, t := range tests {
		importPaths = append(importPaths, t.importPath)
	}
	packages := createPackageMap(importPaths)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotPackageName, gotOk := packages[tt.importPath]
			if gotPackageName != tt.wantPackageName {
				t.Errorf("createPackageMap() gotPackageName = %v, wantPackageName = %v", gotPackageName, tt.wantPackageName)
			}
			if gotOk != tt.wantOK {
				t.Errorf("createPackageMap() gotOk = %v, wantOK = %v", gotOk, tt.wantOK)
			}
		})
	}
}
