package options

import (
	"reflect"
	"strings"
	"testing"

	_ "github.com/docker/libnetwork/testutils"
)

func TestGenerate(t *testing.T) {
	gen := NewGeneric()
	gen["Int"] = 1
	gen["Rune"] = 'b'
	gen["Float64"] = 2.0

	type Model struct {
		Int     int
		Rune    rune
		Float64 float64
	}

	result, err := GenerateFromModel(gen, Model{})

	if err != nil {
		t.Fatal(err)
	}

	cast, ok := result.(Model)
	if !ok {
		t.Fatalf("result has unexpected type %s", reflect.TypeOf(result))
	}
	if expected := 1; cast.Int != expected {
		t.Fatalf("wrong value for field Int: expected %v, got %v", expected, cast.Int)
	}
	if expected := 'b'; cast.Rune != expected {
		t.Fatalf("wrong value for field Rune: expected %v, got %v", expected, cast.Rune)
	}
	if expected := 2.0; cast.Float64 != expected {
		t.Fatalf("wrong value for field Int: expected %v, got %v", expected, cast.Float64)
	}
}

func TestGeneratePtr(t *testing.T) {
	gen := NewGeneric()
	gen["Int"] = 1
	gen["Rune"] = 'b'
	gen["Float64"] = 2.0

	type Model struct {
		Int     int
		Rune    rune
		Float64 float64
	}

	result, err := GenerateFromModel(gen, &Model{})

	if err != nil {
		t.Fatal(err)
	}

	cast, ok := result.(*Model)
	if !ok {
		t.Fatalf("result has unexpected type %s", reflect.TypeOf(result))
	}
	if expected := 1; cast.Int != expected {
		t.Fatalf("wrong value for field Int: expected %v, got %v", expected, cast.Int)
	}
	if expected := 'b'; cast.Rune != expected {
		t.Fatalf("wrong value for field Rune: expected %v, got %v", expected, cast.Rune)
	}
	if expected := 2.0; cast.Float64 != expected {
		t.Fatalf("wrong value for field Int: expected %v, got %v", expected, cast.Float64)
	}
}

func TestGenerateMissingField(t *testing.T) {
	type Model struct{}
	_, err := GenerateFromModel(Generic{"foo": "bar"}, Model{})

	if _, ok := err.(NoSuchFieldError); !ok {
		t.Fatalf("expected NoSuchFieldError, got %#v", err)
	} else if expected := "no field"; !strings.Contains(err.Error(), expected) {
		t.Fatalf("expected %q in error message, got %s", expected, err.Error())
	}
}

func TestFieldCannotBeSet(t *testing.T) {
	type Model struct{ foo int }
	_, err := GenerateFromModel(Generic{"foo": "bar"}, Model{})

	if _, ok := err.(CannotSetFieldError); !ok {
		t.Fatalf("expected CannotSetFieldError, got %#v", err)
	} else if expected := "cannot set field"; !strings.Contains(err.Error(), expected) {
		t.Fatalf("expected %q in error message, got %s", expected, err.Error())
	}
}

func TestTypeMismatchError(t *testing.T) {
	type Model struct{ Foo int }
	_, err := GenerateFromModel(Generic{"Foo": "bar"}, Model{})

	if _, ok := err.(TypeMismatchError); !ok {
		t.Fatalf("expected TypeMismatchError, got %#v", err)
	} else if expected := "type mismatch"; !strings.Contains(err.Error(), expected) {
		t.Fatalf("expected %q in error message, got %s", expected, err.Error())
	}
}
