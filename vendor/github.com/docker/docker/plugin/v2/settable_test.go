package v2

import (
	"reflect"
	"testing"
)

func TestNewSettable(t *testing.T) {
	contexts := []struct {
		arg   string
		name  string
		field string
		value string
		err   error
	}{
		{"name=value", "name", "", "value", nil},
		{"name", "name", "", "", nil},
		{"name.field=value", "name", "field", "value", nil},
		{"name.field", "name", "field", "", nil},
		{"=value", "", "", "", errInvalidFormat},
		{"=", "", "", "", errInvalidFormat},
	}

	for _, c := range contexts {
		s, err := newSettable(c.arg)
		if err != c.err {
			t.Fatalf("expected error to be %v, got %v", c.err, err)
		}

		if s.name != c.name {
			t.Fatalf("expected name to be %q, got %q", c.name, s.name)
		}

		if s.field != c.field {
			t.Fatalf("expected field to be %q, got %q", c.field, s.field)
		}

		if s.value != c.value {
			t.Fatalf("expected value to be %q, got %q", c.value, s.value)
		}

	}
}

func TestIsSettable(t *testing.T) {
	contexts := []struct {
		allowedSettableFields []string
		set                   settable
		settable              []string
		result                bool
		err                   error
	}{
		{allowedSettableFieldsEnv, settable{}, []string{}, false, nil},
		{allowedSettableFieldsEnv, settable{field: "value"}, []string{}, false, nil},
		{allowedSettableFieldsEnv, settable{}, []string{"value"}, true, nil},
		{allowedSettableFieldsEnv, settable{field: "value"}, []string{"value"}, true, nil},
		{allowedSettableFieldsEnv, settable{field: "foo"}, []string{"value"}, false, nil},
		{allowedSettableFieldsEnv, settable{field: "foo"}, []string{"foo"}, false, nil},
		{allowedSettableFieldsEnv, settable{}, []string{"value1", "value2"}, false, errMultipleFields},
	}

	for _, c := range contexts {
		if res, err := c.set.isSettable(c.allowedSettableFields, c.settable); res != c.result {
			t.Fatalf("expected result to be %t, got %t", c.result, res)
		} else if err != c.err {
			t.Fatalf("expected error to be %v, got %v", c.err, err)
		}
	}
}

func TestUpdateSettingsEnv(t *testing.T) {
	contexts := []struct {
		env    []string
		set    settable
		newEnv []string
	}{
		{[]string{}, settable{name: "DEBUG", value: "1"}, []string{"DEBUG=1"}},
		{[]string{"DEBUG=0"}, settable{name: "DEBUG", value: "1"}, []string{"DEBUG=1"}},
		{[]string{"FOO=0"}, settable{name: "DEBUG", value: "1"}, []string{"FOO=0", "DEBUG=1"}},
		{[]string{"FOO=0", "DEBUG=0"}, settable{name: "DEBUG", value: "1"}, []string{"FOO=0", "DEBUG=1"}},
		{[]string{"FOO=0", "DEBUG=0", "BAR=1"}, settable{name: "DEBUG", value: "1"}, []string{"FOO=0", "DEBUG=1", "BAR=1"}},
	}

	for _, c := range contexts {
		updateSettingsEnv(&c.env, &c.set)

		if !reflect.DeepEqual(c.env, c.newEnv) {
			t.Fatalf("expected env to be %q, got %q", c.newEnv, c.env)
		}
	}
}
