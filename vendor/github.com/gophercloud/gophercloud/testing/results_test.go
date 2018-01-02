package testing

import (
	"encoding/json"
	"testing"

	"github.com/gophercloud/gophercloud"
	th "github.com/gophercloud/gophercloud/testhelper"
)

var singleResponse = `
{
	"person": {
		"name": "Bill",
		"email": "bill@example.com",
		"location": "Canada"
	}
}
`

var multiResponse = `
{
	"people": [
		{
			"name": "Bill",
			"email": "bill@example.com",
			"location": "Canada"
		},
		{
			"name": "Ted",
			"email": "ted@example.com",
			"location": "Mexico"
		}
	]
}
`

type TestPerson struct {
	Name  string `json:"-"`
	Email string `json:"email"`
}

func (r *TestPerson) UnmarshalJSON(b []byte) error {
	type tmp TestPerson
	var s struct {
		tmp
		Name string `json:"name"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = TestPerson(s.tmp)
	r.Name = s.Name + " unmarshalled"

	return nil
}

type TestPersonExt struct {
	Location string `json:"-"`
}

func (r *TestPersonExt) UnmarshalJSON(b []byte) error {
	type tmp TestPersonExt
	var s struct {
		tmp
		Location string `json:"location"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = TestPersonExt(s.tmp)
	r.Location = s.Location + " unmarshalled"

	return nil
}

type TestPersonWithExtensions struct {
	TestPerson
	TestPersonExt
}

type TestPersonWithExtensionsNamed struct {
	TestPerson    TestPerson
	TestPersonExt TestPersonExt
}

// TestUnmarshalAnonymousStruct tests if UnmarshalJSON is called on each
// of the anonymous structs contained in an overarching struct.
func TestUnmarshalAnonymousStructs(t *testing.T) {
	var actual TestPersonWithExtensions

	var dejson interface{}
	sejson := []byte(singleResponse)
	err := json.Unmarshal(sejson, &dejson)
	if err != nil {
		t.Fatal(err)
	}

	var singleResult = gophercloud.Result{
		Body: dejson,
	}

	err = singleResult.ExtractIntoStructPtr(&actual, "person")
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "Bill unmarshalled", actual.Name)
	th.AssertEquals(t, "Canada unmarshalled", actual.Location)
}

// TestUnmarshalSliceofAnonymousStructs tests if UnmarshalJSON is called on each
// of the anonymous structs contained in an overarching struct slice.
func TestUnmarshalSliceOfAnonymousStructs(t *testing.T) {
	var actual []TestPersonWithExtensions

	var dejson interface{}
	sejson := []byte(multiResponse)
	err := json.Unmarshal(sejson, &dejson)
	if err != nil {
		t.Fatal(err)
	}

	var multiResult = gophercloud.Result{
		Body: dejson,
	}

	err = multiResult.ExtractIntoSlicePtr(&actual, "people")
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "Bill unmarshalled", actual[0].Name)
	th.AssertEquals(t, "Canada unmarshalled", actual[0].Location)
	th.AssertEquals(t, "Ted unmarshalled", actual[1].Name)
	th.AssertEquals(t, "Mexico unmarshalled", actual[1].Location)
}

// TestUnmarshalSliceOfStruct tests if extracting results from a "normal"
// struct still works correctly.
func TestUnmarshalSliceofStruct(t *testing.T) {
	var actual []TestPerson

	var dejson interface{}
	sejson := []byte(multiResponse)
	err := json.Unmarshal(sejson, &dejson)
	if err != nil {
		t.Fatal(err)
	}

	var multiResult = gophercloud.Result{
		Body: dejson,
	}

	err = multiResult.ExtractIntoSlicePtr(&actual, "people")
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "Bill unmarshalled", actual[0].Name)
	th.AssertEquals(t, "Ted unmarshalled", actual[1].Name)
}

// TestUnmarshalNamedStruct tests if the result is empty.
func TestUnmarshalNamedStructs(t *testing.T) {
	var actual TestPersonWithExtensionsNamed

	var dejson interface{}
	sejson := []byte(singleResponse)
	err := json.Unmarshal(sejson, &dejson)
	if err != nil {
		t.Fatal(err)
	}

	var singleResult = gophercloud.Result{
		Body: dejson,
	}

	err = singleResult.ExtractIntoStructPtr(&actual, "person")
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "", actual.TestPerson.Name)
	th.AssertEquals(t, "", actual.TestPersonExt.Location)
}

// TestUnmarshalSliceofNamedStructs tests if the result is empty.
func TestUnmarshalSliceOfNamedStructs(t *testing.T) {
	var actual []TestPersonWithExtensionsNamed

	var dejson interface{}
	sejson := []byte(multiResponse)
	err := json.Unmarshal(sejson, &dejson)
	if err != nil {
		t.Fatal(err)
	}

	var multiResult = gophercloud.Result{
		Body: dejson,
	}

	err = multiResult.ExtractIntoSlicePtr(&actual, "people")
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "", actual[0].TestPerson.Name)
	th.AssertEquals(t, "", actual[0].TestPersonExt.Location)
	th.AssertEquals(t, "", actual[1].TestPerson.Name)
	th.AssertEquals(t, "", actual[1].TestPersonExt.Location)
}
