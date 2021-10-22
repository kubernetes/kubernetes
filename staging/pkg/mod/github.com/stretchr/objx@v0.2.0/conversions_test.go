package objx_test

import (
	"net/url"
	"testing"

	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConversionJSON(t *testing.T) {
	jsonString := `{"name":"Mat"}`
	o := objx.MustFromJSON(jsonString)

	result, err := o.JSON()

	require.NoError(t, err)
	assert.Equal(t, jsonString, result)
	assert.Equal(t, jsonString, o.MustJSON())

	i := objx.Map{
		"a": map[interface{}]interface{}{"b": objx.Map{"c": map[interface{}]interface{}{"d": "e"}},
			"f": []objx.Map{{"g": map[interface{}]interface{}{"h": "i"}}},
			"j": []map[string]interface{}{{"k": map[interface{}]interface{}{"l": "m"}}},
			"n": []interface{}{objx.Map{"o": "p"}},
		},
	}

	jsonString = `{"a":{"b":{"c":{"d":"e"}},"f":[{"g":{"h":"i"}}],"j":[{"k":{"l":"m"}}],"n":[{"o":"p"}]}}`
	result, err = i.JSON()
	require.NoError(t, err)
	assert.Equal(t, jsonString, result)
	assert.Equal(t, jsonString, i.MustJSON())
}

func TestConversionJSONWithError(t *testing.T) {
	o := objx.MSI()
	o["test"] = func() {}

	assert.Panics(t, func() {
		o.MustJSON()
	})

	_, err := o.JSON()

	assert.Error(t, err)
}

func TestConversionBase64(t *testing.T) {
	o := objx.Map{"name": "Mat"}

	result, err := o.Base64()

	require.NoError(t, err)
	assert.Equal(t, "eyJuYW1lIjoiTWF0In0=", result)
	assert.Equal(t, "eyJuYW1lIjoiTWF0In0=", o.MustBase64())
}

func TestConversionBase64WithError(t *testing.T) {
	o := objx.MSI()
	o["test"] = func() {}

	assert.Panics(t, func() {
		o.MustBase64()
	})

	_, err := o.Base64()

	assert.Error(t, err)
}

func TestConversionSignedBase64(t *testing.T) {
	o := objx.Map{"name": "Mat"}

	result, err := o.SignedBase64("key")

	require.NoError(t, err)
	assert.Equal(t, "eyJuYW1lIjoiTWF0In0=_67ee82916f90b2c0d68c903266e8998c9ef0c3d6", result)
	assert.Equal(t, "eyJuYW1lIjoiTWF0In0=_67ee82916f90b2c0d68c903266e8998c9ef0c3d6", o.MustSignedBase64("key"))
}

func TestConversionSignedBase64WithError(t *testing.T) {
	o := objx.MSI()
	o["test"] = func() {}

	assert.Panics(t, func() {
		o.MustSignedBase64("key")
	})

	_, err := o.SignedBase64("key")

	assert.Error(t, err)
}

func TestConversionURLValues(t *testing.T) {
	m := getURLQueryMap()
	u := m.URLValues()

	assert.Equal(t, url.Values{
		"abc":                []string{"123"},
		"name":               []string{"Mat"},
		"data[age]":          []string{"30"},
		"data[height]":       []string{"162"},
		"data[arr][]":        []string{"1", "2"},
		"stats[]":            []string{"1", "2"},
		"bools[]":            []string{"true", "false"},
		"mapSlice[][age]":    []string{"40"},
		"mapSlice[][height]": []string{"152"},
		"msiData[age]":       []string{"30"},
		"msiData[height]":    []string{"162"},
		"msiData[arr][]":     []string{"1", "2"},
		"msiSlice[][age]":    []string{"40"},
		"msiSlice[][height]": []string{"152"},
	}, u)
}

func TestConversionURLQuery(t *testing.T) {
	m := getURLQueryMap()
	u, err := m.URLQuery()

	assert.Nil(t, err)
	require.NotNil(t, u)

	ue, err := url.QueryUnescape(u)
	assert.Nil(t, err)
	require.NotNil(t, ue)

	assert.Equal(t, "abc=123&bools[]=true&bools[]=false&data[age]=30&data[arr][]=1&data[arr][]=2&data[height]=162&mapSlice[][age]=40&mapSlice[][height]=152&msiData[age]=30&msiData[arr][]=1&msiData[arr][]=2&msiData[height]=162&msiSlice[][age]=40&msiSlice[][height]=152&name=Mat&stats[]=1&stats[]=2", ue)
}

func TestConversionURLQueryNoSliceKeySuffix(t *testing.T) {
	m := getURLQueryMap()
	objx.SetURLValuesSliceKeySuffix(objx.URLValuesSliceKeySuffixEmpty)
	u, err := m.URLQuery()

	assert.Nil(t, err)
	require.NotNil(t, u)

	ue, err := url.QueryUnescape(u)
	assert.Nil(t, err)
	require.NotNil(t, ue)

	assert.Equal(t, "abc=123&bools=true&bools=false&data[age]=30&data[arr]=1&data[arr]=2&data[height]=162&mapSlice[age]=40&mapSlice[height]=152&msiData[age]=30&msiData[arr]=1&msiData[arr]=2&msiData[height]=162&msiSlice[age]=40&msiSlice[height]=152&name=Mat&stats=1&stats=2", ue)
}

func TestConversionURLQueryIndexSliceKeySuffix(t *testing.T) {
	m := getURLQueryMap()
	m.Set("mapSlice", []objx.Map{{"age": 40, "sex": "male"}, {"height": 152}})
	objx.SetURLValuesSliceKeySuffix(objx.URLValuesSliceKeySuffixIndex)
	u, err := m.URLQuery()

	assert.Nil(t, err)
	require.NotNil(t, u)

	ue, err := url.QueryUnescape(u)
	assert.Nil(t, err)
	require.NotNil(t, ue)

	assert.Equal(t, "abc=123&bools[0]=true&bools[1]=false&data[age]=30&data[arr][0]=1&data[arr][1]=2&data[height]=162&mapSlice[0][age]=40&mapSlice[0][sex]=male&mapSlice[1][height]=152&msiData[age]=30&msiData[arr][0]=1&msiData[arr][1]=2&msiData[height]=162&msiSlice[0][age]=40&msiSlice[1][height]=152&name=Mat&stats[0]=1&stats[1]=2", ue)
}

func TestValidityURLQuerySliceKeySuffix(t *testing.T) {
	err := objx.SetURLValuesSliceKeySuffix("")
	assert.Nil(t, err)
	err = objx.SetURLValuesSliceKeySuffix("[]")
	assert.Nil(t, err)
	err = objx.SetURLValuesSliceKeySuffix("[i]")
	assert.Nil(t, err)
	err = objx.SetURLValuesSliceKeySuffix("{}")
	assert.Error(t, err)
}

func getURLQueryMap() objx.Map {
	return objx.Map{
		"abc":      123,
		"name":     "Mat",
		"data":     objx.Map{"age": 30, "height": 162, "arr": []int{1, 2}},
		"mapSlice": []objx.Map{{"age": 40}, {"height": 152}},
		"msiData":  map[string]interface{}{"age": 30, "height": 162, "arr": []int{1, 2}},
		"msiSlice": []map[string]interface{}{{"age": 40}, {"height": 152}},
		"stats":    []string{"1", "2"},
		"bools":    []bool{true, false},
	}
}
