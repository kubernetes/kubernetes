package testing

import (
	"net/url"
	"reflect"
	"testing"

	"github.com/gophercloud/gophercloud"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestMaybeString(t *testing.T) {
	testString := ""
	var expected *string
	actual := gophercloud.MaybeString(testString)
	th.CheckDeepEquals(t, expected, actual)

	testString = "carol"
	expected = &testString
	actual = gophercloud.MaybeString(testString)
	th.CheckDeepEquals(t, expected, actual)
}

func TestMaybeInt(t *testing.T) {
	testInt := 0
	var expected *int
	actual := gophercloud.MaybeInt(testInt)
	th.CheckDeepEquals(t, expected, actual)

	testInt = 4
	expected = &testInt
	actual = gophercloud.MaybeInt(testInt)
	th.CheckDeepEquals(t, expected, actual)
}

func TestBuildQueryString(t *testing.T) {
	type testVar string
	iFalse := false
	opts := struct {
		J  int               `q:"j"`
		R  string            `q:"r,required"`
		C  bool              `q:"c"`
		S  []string          `q:"s"`
		TS []testVar         `q:"ts"`
		TI []int             `q:"ti"`
		F  *bool             `q:"f"`
		M  map[string]string `q:"m"`
	}{
		J:  2,
		R:  "red",
		C:  true,
		S:  []string{"one", "two", "three"},
		TS: []testVar{"a", "b"},
		TI: []int{1, 2},
		F:  &iFalse,
		M:  map[string]string{"k1": "success1"},
	}
	expected := &url.URL{RawQuery: "c=true&f=false&j=2&m=%7B%27k1%27%3A%27success1%27%7D&r=red&s=one&s=two&s=three&ti=1&ti=2&ts=a&ts=b"}
	actual, err := gophercloud.BuildQueryString(&opts)
	if err != nil {
		t.Errorf("Error building query string: %v", err)
	}
	th.CheckDeepEquals(t, expected, actual)

	opts = struct {
		J  int               `q:"j"`
		R  string            `q:"r,required"`
		C  bool              `q:"c"`
		S  []string          `q:"s"`
		TS []testVar         `q:"ts"`
		TI []int             `q:"ti"`
		F  *bool             `q:"f"`
		M  map[string]string `q:"m"`
	}{
		J: 2,
		C: true,
	}
	_, err = gophercloud.BuildQueryString(&opts)
	if err == nil {
		t.Errorf("Expected error: 'Required field not set'")
	}
	th.CheckDeepEquals(t, expected, actual)

	_, err = gophercloud.BuildQueryString(map[string]interface{}{"Number": 4})
	if err == nil {
		t.Errorf("Expected error: 'Options type is not a struct'")
	}
}

func TestBuildHeaders(t *testing.T) {
	testStruct := struct {
		Accept string `h:"Accept"`
		Num    int    `h:"Number,required"`
		Style  bool   `h:"Style"`
	}{
		Accept: "application/json",
		Num:    4,
		Style:  true,
	}
	expected := map[string]string{"Accept": "application/json", "Number": "4", "Style": "true"}
	actual, err := gophercloud.BuildHeaders(&testStruct)
	th.CheckNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)

	testStruct.Num = 0
	_, err = gophercloud.BuildHeaders(&testStruct)
	if err == nil {
		t.Errorf("Expected error: 'Required header not set'")
	}

	_, err = gophercloud.BuildHeaders(map[string]interface{}{"Number": 4})
	if err == nil {
		t.Errorf("Expected error: 'Options type is not a struct'")
	}
}

func TestQueriesAreEscaped(t *testing.T) {
	type foo struct {
		Name  string `q:"something"`
		Shape string `q:"else"`
	}

	expected := &url.URL{RawQuery: "else=Triangl+e&something=blah%2B%3F%21%21foo"}

	actual, err := gophercloud.BuildQueryString(foo{Name: "blah+?!!foo", Shape: "Triangl e"})
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, expected, actual)
}

func TestBuildRequestBody(t *testing.T) {
	type PasswordCredentials struct {
		Username string `json:"username" required:"true"`
		Password string `json:"password" required:"true"`
	}

	type TokenCredentials struct {
		ID string `json:"id,omitempty" required:"true"`
	}

	type orFields struct {
		Filler int `json:"filler,omitempty"`
		F1     int `json:"f1,omitempty" or:"F2"`
		F2     int `json:"f2,omitempty" or:"F1"`
	}

	// AuthOptions wraps a gophercloud AuthOptions in order to adhere to the AuthOptionsBuilder
	// interface.
	type AuthOptions struct {
		PasswordCredentials *PasswordCredentials `json:"passwordCredentials,omitempty" xor:"TokenCredentials"`

		// The TenantID and TenantName fields are optional for the Identity V2 API.
		// Some providers allow you to specify a TenantName instead of the TenantId.
		// Some require both. Your provider's authentication policies will determine
		// how these fields influence authentication.
		TenantID   string `json:"tenantId,omitempty"`
		TenantName string `json:"tenantName,omitempty"`

		// TokenCredentials allows users to authenticate (possibly as another user) with an
		// authentication token ID.
		TokenCredentials *TokenCredentials `json:"token,omitempty" xor:"PasswordCredentials"`

		OrFields *orFields `json:"or_fields,omitempty"`
	}

	var successCases = []struct {
		opts     AuthOptions
		expected map[string]interface{}
	}{
		{
			AuthOptions{
				PasswordCredentials: &PasswordCredentials{
					Username: "me",
					Password: "swordfish",
				},
			},
			map[string]interface{}{
				"auth": map[string]interface{}{
					"passwordCredentials": map[string]interface{}{
						"password": "swordfish",
						"username": "me",
					},
				},
			},
		},
		{
			AuthOptions{
				TokenCredentials: &TokenCredentials{
					ID: "1234567",
				},
			},
			map[string]interface{}{
				"auth": map[string]interface{}{
					"token": map[string]interface{}{
						"id": "1234567",
					},
				},
			},
		},
	}

	for _, successCase := range successCases {
		actual, err := gophercloud.BuildRequestBody(successCase.opts, "auth")
		th.AssertNoErr(t, err)
		th.AssertDeepEquals(t, successCase.expected, actual)
	}

	var failCases = []struct {
		opts     AuthOptions
		expected error
	}{
		{
			AuthOptions{
				TenantID:   "987654321",
				TenantName: "me",
			},
			gophercloud.ErrMissingInput{},
		},
		{
			AuthOptions{
				TokenCredentials: &TokenCredentials{
					ID: "1234567",
				},
				PasswordCredentials: &PasswordCredentials{
					Username: "me",
					Password: "swordfish",
				},
			},
			gophercloud.ErrMissingInput{},
		},
		{
			AuthOptions{
				PasswordCredentials: &PasswordCredentials{
					Password: "swordfish",
				},
			},
			gophercloud.ErrMissingInput{},
		},
		{
			AuthOptions{
				PasswordCredentials: &PasswordCredentials{
					Username: "me",
					Password: "swordfish",
				},
				OrFields: &orFields{
					Filler: 2,
				},
			},
			gophercloud.ErrMissingInput{},
		},
	}

	for _, failCase := range failCases {
		_, err := gophercloud.BuildRequestBody(failCase.opts, "auth")
		th.AssertDeepEquals(t, reflect.TypeOf(failCase.expected), reflect.TypeOf(err))
	}
}
