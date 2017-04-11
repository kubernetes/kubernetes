package errcode

import (
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

// TestErrorsManagement does a quick check of the Errors type to ensure that
// members are properly pushed and marshaled.
var ErrorCodeTest1 = Register("test.errors", ErrorDescriptor{
	Value:          "TEST1",
	Message:        "test error 1",
	Description:    `Just a test message #1.`,
	HTTPStatusCode: http.StatusInternalServerError,
})

var ErrorCodeTest2 = Register("test.errors", ErrorDescriptor{
	Value:          "TEST2",
	Message:        "test error 2",
	Description:    `Just a test message #2.`,
	HTTPStatusCode: http.StatusNotFound,
})

var ErrorCodeTest3 = Register("test.errors", ErrorDescriptor{
	Value:          "TEST3",
	Message:        "Sorry %q isn't valid",
	Description:    `Just a test message #3.`,
	HTTPStatusCode: http.StatusNotFound,
})

// TestErrorCodes ensures that error code format, mappings and
// marshaling/unmarshaling. round trips are stable.
func TestErrorCodes(t *testing.T) {
	if len(errorCodeToDescriptors) == 0 {
		t.Fatal("errors aren't loaded!")
	}

	for ec, desc := range errorCodeToDescriptors {
		if ec != desc.Code {
			t.Fatalf("error code in descriptor isn't correct, %q != %q", ec, desc.Code)
		}

		if idToDescriptors[desc.Value].Code != ec {
			t.Fatalf("error code in idToDesc isn't correct, %q != %q", idToDescriptors[desc.Value].Code, ec)
		}

		if ec.Message() != desc.Message {
			t.Fatalf("ec.Message doesn't mtach desc.Message: %q != %q", ec.Message(), desc.Message)
		}

		// Test (de)serializing the ErrorCode
		p, err := json.Marshal(ec)
		if err != nil {
			t.Fatalf("couldn't marshal ec %v: %v", ec, err)
		}

		if len(p) <= 0 {
			t.Fatalf("expected content in marshaled before for error code %v", ec)
		}

		// First, unmarshal to interface and ensure we have a string.
		var ecUnspecified interface{}
		if err := json.Unmarshal(p, &ecUnspecified); err != nil {
			t.Fatalf("error unmarshaling error code %v: %v", ec, err)
		}

		if _, ok := ecUnspecified.(string); !ok {
			t.Fatalf("expected a string for error code %v on unmarshal got a %T", ec, ecUnspecified)
		}

		// Now, unmarshal with the error code type and ensure they are equal
		var ecUnmarshaled ErrorCode
		if err := json.Unmarshal(p, &ecUnmarshaled); err != nil {
			t.Fatalf("error unmarshaling error code %v: %v", ec, err)
		}

		if ecUnmarshaled != ec {
			t.Fatalf("unexpected error code during error code marshal/unmarshal: %v != %v", ecUnmarshaled, ec)
		}

		expectedErrorString := strings.ToLower(strings.Replace(ec.Descriptor().Value, "_", " ", -1))
		if ec.Error() != expectedErrorString {
			t.Fatalf("unexpected return from %v.Error(): %q != %q", ec, ec.Error(), expectedErrorString)
		}
	}

}

func TestErrorsManagement(t *testing.T) {
	var errs Errors

	errs = append(errs, ErrorCodeTest1)
	errs = append(errs, ErrorCodeTest2.WithDetail(
		map[string]interface{}{"digest": "sometestblobsumdoesntmatter"}))
	errs = append(errs, ErrorCodeTest3.WithArgs("BOOGIE"))
	errs = append(errs, ErrorCodeTest3.WithArgs("BOOGIE").WithDetail("data"))

	p, err := json.Marshal(errs)

	if err != nil {
		t.Fatalf("error marashaling errors: %v", err)
	}

	expectedJSON := `{"errors":[` +
		`{"code":"TEST1","message":"test error 1"},` +
		`{"code":"TEST2","message":"test error 2","detail":{"digest":"sometestblobsumdoesntmatter"}},` +
		`{"code":"TEST3","message":"Sorry \"BOOGIE\" isn't valid"},` +
		`{"code":"TEST3","message":"Sorry \"BOOGIE\" isn't valid","detail":"data"}` +
		`]}`

	if string(p) != expectedJSON {
		t.Fatalf("unexpected json:\ngot:\n%q\n\nexpected:\n%q", string(p), expectedJSON)
	}

	// Now test the reverse
	var unmarshaled Errors
	if err := json.Unmarshal(p, &unmarshaled); err != nil {
		t.Fatalf("unexpected error unmarshaling error envelope: %v", err)
	}

	if !reflect.DeepEqual(unmarshaled, errs) {
		t.Fatalf("errors not equal after round trip:\nunmarshaled:\n%#v\n\nerrs:\n%#v", unmarshaled, errs)
	}

	// Test the arg substitution stuff
	e1 := unmarshaled[3].(Error)
	exp1 := `Sorry "BOOGIE" isn't valid`
	if e1.Message != exp1 {
		t.Fatalf("Wrong msg, got:\n%q\n\nexpected:\n%q", e1.Message, exp1)
	}

	exp1 = "test3: " + exp1
	if e1.Error() != exp1 {
		t.Fatalf("Error() didn't return the right string, got:%s\nexpected:%s", e1.Error(), exp1)
	}

	// Test again with a single value this time
	errs = Errors{ErrorCodeUnknown}
	expectedJSON = "{\"errors\":[{\"code\":\"UNKNOWN\",\"message\":\"unknown error\"}]}"
	p, err = json.Marshal(errs)

	if err != nil {
		t.Fatalf("error marashaling errors: %v", err)
	}

	if string(p) != expectedJSON {
		t.Fatalf("unexpected json: %q != %q", string(p), expectedJSON)
	}

	// Now test the reverse
	unmarshaled = nil
	if err := json.Unmarshal(p, &unmarshaled); err != nil {
		t.Fatalf("unexpected error unmarshaling error envelope: %v", err)
	}

	if !reflect.DeepEqual(unmarshaled, errs) {
		t.Fatalf("errors not equal after round trip:\nunmarshaled:\n%#v\n\nerrs:\n%#v", unmarshaled, errs)
	}

	// Verify that calling WithArgs() more than once does the right thing.
	// Meaning creates a new Error and uses the ErrorCode Message
	e1 = ErrorCodeTest3.WithArgs("test1")
	e2 := e1.WithArgs("test2")
	if &e1 == &e2 {
		t.Fatalf("args: e2 and e1 should not be the same, but they are")
	}
	if e2.Message != `Sorry "test2" isn't valid` {
		t.Fatalf("e2 had wrong message: %q", e2.Message)
	}

	// Verify that calling WithDetail() more than once does the right thing.
	// Meaning creates a new Error and overwrites the old detail field
	e1 = ErrorCodeTest3.WithDetail("stuff1")
	e2 = e1.WithDetail("stuff2")
	if &e1 == &e2 {
		t.Fatalf("detail: e2 and e1 should not be the same, but they are")
	}
	if e2.Detail != `stuff2` {
		t.Fatalf("e2 had wrong detail: %q", e2.Detail)
	}

}
