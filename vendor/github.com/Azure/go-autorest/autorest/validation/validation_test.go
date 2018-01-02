package validation

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCheckForUniqueInArrayTrue(t *testing.T) {
	require.Equal(t, checkForUniqueInArray(reflect.ValueOf([]int{1, 2, 3})), true)
}

func TestCheckForUniqueInArrayFalse(t *testing.T) {
	require.Equal(t, checkForUniqueInArray(reflect.ValueOf([]int{1, 2, 3, 3})), false)
}

func TestCheckForUniqueInArrayEmpty(t *testing.T) {
	require.Equal(t, checkForUniqueInArray(reflect.ValueOf([]int{})), false)
}

func TestCheckForUniqueInMapTrue(t *testing.T) {
	require.Equal(t, checkForUniqueInMap(reflect.ValueOf(map[string]int{"one": 1, "two": 2})), true)
}

func TestCheckForUniqueInMapFalse(t *testing.T) {
	require.Equal(t, checkForUniqueInMap(reflect.ValueOf(map[int]string{1: "one", 2: "one"})), false)
}

func TestCheckForUniqueInMapEmpty(t *testing.T) {
	require.Equal(t, checkForUniqueInMap(reflect.ValueOf(map[int]string{})), false)
}

func TestCheckEmpty_WithValueEmptyRuleTrue(t *testing.T) {
	var x interface{}
	v := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   true,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), v, "value can not be null or empty; required parameter")
	require.Equal(t, checkEmpty(reflect.ValueOf(x), v).Error(), expected.Error())
}

func TestCheckEmpty_WithEmptyStringRuleFalse(t *testing.T) {
	var x interface{}
	v := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   false,
		Chain:  nil,
	}
	require.Nil(t, checkEmpty(reflect.ValueOf(x), v))
}

func TestCheckEmpty_IncorrectRule(t *testing.T) {
	var x interface{}
	v := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   10,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), v, fmt.Sprintf("rule must be bool value for %v constraint; got: %v", v.Name, v.Rule))
	require.Equal(t, checkEmpty(reflect.ValueOf(x), v).Error(), expected.Error())
}

func TestCheckEmpty_WithErrorArray(t *testing.T) {
	var x interface{} = []string{}
	v := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   true,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), v, "value can not be null or empty; required parameter")
	require.Equal(t, checkEmpty(reflect.ValueOf(x), v).Error(), expected.Error())
}

func TestCheckNil_WithNilValueRuleTrue(t *testing.T) {
	var x interface{}
	v := Constraint{
		Target: "x",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{"x", MaxItems, 4, nil},
		},
	}
	expected := createError(reflect.ValueOf(x), v, "value can not be null; required parameter")
	require.Equal(t, checkNil(reflect.ValueOf(x), v).Error(), expected.Error())
}

func TestCheckNil_WithNilValueRuleFalse(t *testing.T) {
	var x interface{}
	v := Constraint{
		Target: "x",
		Name:   Null,
		Rule:   false,
		Chain: []Constraint{
			{"x", MaxItems, 4, nil},
		},
	}
	require.Nil(t, checkNil(reflect.ValueOf(x), v))
}

func TestCheckNil_IncorrectRule(t *testing.T) {
	var x interface{}
	c := Constraint{
		Target: "str",
		Name:   Null,
		Rule:   10,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("rule must be bool value for %v constraint; got: %v", c.Name, c.Rule))
	require.Equal(t, checkNil(reflect.ValueOf(x), c).Error(), expected.Error())
}

func TestValidateArrayMap_WithNilValueRuleTrue(t *testing.T) {
	var a []string
	var x interface{} = a
	c := Constraint{
		Target: "arr",
		Name:   Null,
		Rule:   true,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, "value can not be null; required parameter")
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c), expected)
}

func TestValidateArrayMap_WithNilValueRuleFalse(t *testing.T) {
	var x interface{} = []string{}
	c := Constraint{
		Target: "arr",
		Name:   Null,
		Rule:   false,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_WithValueRuleNullTrue(t *testing.T) {
	var x interface{} = []string{"1", "2"}
	c := Constraint{
		Target: "arr",
		Name:   Null,
		Rule:   false,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_WithEmptyValueRuleTrue(t *testing.T) {
	var x interface{} = []string{}
	c := Constraint{
		Target: "arr",
		Name:   Empty,
		Rule:   true,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, "value can not be null or empty; required parameter")
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c), expected)
}

func TestValidateArrayMap_WithEmptyValueRuleFalse(t *testing.T) {
	var x interface{} = []string{}
	c := Constraint{
		Target: "arr",
		Name:   Empty,
		Rule:   false,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_WithEmptyRuleEmptyTrue(t *testing.T) {
	var x interface{} = []string{"1", "2"}
	c := Constraint{
		Target: "arr",
		Name:   Empty,
		Rule:   false,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_MaxItemsIncorrectRule(t *testing.T) {
	var x interface{} = []string{"1", "2"}
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   false,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("rule must be integer for %v constraint; got: %v", c.Name, c.Rule))
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c).Error(), expected.Error())
}

func TestValidateArrayMap_MaxItemsNoError(t *testing.T) {
	var x interface{} = []string{"1", "2"}
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_MaxItemsWithError(t *testing.T) {
	var x interface{} = []string{"1", "2", "3"}
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   2,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("maximum item limit is %v; got: 3", c.Rule))
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c).Error(), expected.Error())
}

func TestValidateArrayMap_MaxItemsWithEmpty(t *testing.T) {
	var x interface{} = []string{}
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_MinItemsIncorrectRule(t *testing.T) {
	var x interface{} = []int{1, 2}
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   false,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("rule must be integer for %v constraint; got: %v", c.Name, c.Rule))
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c).Error(), expected.Error())
}

func TestValidateArrayMap_MinItemsNoError1(t *testing.T) {
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf([]int{1, 2}), c))
}

func TestValidateArrayMap_MinItemsNoError2(t *testing.T) {
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf([]int{1, 2, 3}), c))
}

func TestValidateArrayMap_MinItemsWithError(t *testing.T) {
	var x interface{} = []int{1}
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   2,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("minimum item limit is %v; got: 1", c.Rule))
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c).Error(), expected.Error())
}

func TestValidateArrayMap_MinItemsWithEmpty(t *testing.T) {
	var x interface{} = []int{}
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   2,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("minimum item limit is %v; got: 0", c.Rule))
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c).Error(), expected.Error())
}

func TestValidateArrayMap_Map_MaxItemsIncorrectRule(t *testing.T) {
	var x interface{} = map[int]string{1: "1", 2: "2"}
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   false,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateArrayMap(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("rule must be integer for %v constraint; got: %v", c.Name, c.Rule)), true)
}

func TestValidateArrayMap_Map_MaxItemsNoError(t *testing.T) {
	var x interface{} = map[int]string{1: "1", 2: "2"}
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_Map_MaxItemsWithError(t *testing.T) {
	a := map[int]string{1: "1", 2: "2", 3: "3"}
	var x interface{} = a
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateArrayMap(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("maximum item limit is %v; got: %v", c.Rule, len(a))), true)
}

func TestValidateArrayMap_Map_MaxItemsWithEmpty(t *testing.T) {
	a := map[int]string{}
	var x interface{} = a
	c := Constraint{
		Target: "arr",
		Name:   MaxItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_Map_MinItemsIncorrectRule(t *testing.T) {
	var x interface{} = map[int]string{1: "1", 2: "2"}
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   false,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateArrayMap(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("rule must be integer for %v constraint; got: %v", c.Name, c.Rule)), true)
}

func TestValidateArrayMap_Map_MinItemsNoError1(t *testing.T) {
	var x interface{} = map[int]string{1: "1", 2: "2"}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x),
		Constraint{
			Target: "arr",
			Name:   MinItems,
			Rule:   2,
			Chain:  nil,
		}))
}

func TestValidateArrayMap_Map_MinItemsNoError2(t *testing.T) {
	var x interface{} = map[int]string{1: "1", 2: "2", 3: "3"}
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_Map_MinItemsWithError(t *testing.T) {
	a := map[int]string{1: "1"}
	var x interface{} = a
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   2,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("minimum item limit is %v; got: %v", c.Rule, len(a)))
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c).Error(), expected.Error())
}

func TestValidateArrayMap_Map_MinItemsWithEmpty(t *testing.T) {
	a := map[int]string{}
	var x interface{} = a
	c := Constraint{
		Target: "arr",
		Name:   MinItems,
		Rule:   2,
		Chain:  nil,
	}
	expected := createError(reflect.ValueOf(x), c, fmt.Sprintf("minimum item limit is %v; got: %v", c.Rule, len(a)))
	require.Equal(t, validateArrayMap(reflect.ValueOf(x), c).Error(), expected.Error())
}

// func TestValidateArrayMap_Map_MinItemsNil(t *testing.T) {
// 	var a map[int]float64
// 	var x interface{} = a
// 	c := Constraint{
// 		Target: "str",
// 		Name:   MinItems,
// 		Rule:   true,
// 		Chain:  nil,
// 	}
// 	expected := createError(reflect.Value(x), c, fmt.Sprintf("all items in parameter %v must be unique; got:%v", c.Target, x))
// 	if z := validateArrayMap(reflect.ValueOf(x), c); strings.Contains(z.Error(), "all items in parameter str must be unique;") {
// 		t.Fatalf("autorest/validation: valiateArrayMap failed to return error \nexpect: %v;\ngot: %v", expected, z)
// 	}
// }

func TestValidateArrayMap_Map_UniqueItemsTrue(t *testing.T) {
	var x interface{} = map[float64]int{1.2: 1, 1.4: 2}
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_Map_UniqueItemsFalse(t *testing.T) {
	var x interface{} = map[string]string{"1": "1", "2": "2", "3": "1"}
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("all items in parameter %q must be unique", c.Target)), true)
}

func TestValidateArrayMap_Map_UniqueItemsEmpty(t *testing.T) {
	// Consider Empty map as not unique returns false
	var x interface{} = map[int]float64{}
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("all items in parameter %q must be unique", c.Target)), true)
}

func TestValidateArrayMap_Map_UniqueItemsNil(t *testing.T) {
	var a map[int]float64
	var x interface{} = a
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("all items in parameter %q must be unique; got:%v", c.Target, x)), true)
}

func TestValidateArrayMap_Array_UniqueItemsTrue(t *testing.T) {
	var x interface{} = []int{1, 2}
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_Array_UniqueItemsFalse(t *testing.T) {
	var x interface{} = []string{"1", "2", "1"}
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("all items in parameter %q must be unique; got:%v", c.Target, x)), true)
}

func TestValidateArrayMap_Array_UniqueItemsEmpty(t *testing.T) {
	// Consider Empty array as not unique returns false
	var x interface{} = []float64{}
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("all items in parameter %q must be unique; got:%v", c.Target, x)), true)
}

func TestValidateArrayMap_Array_UniqueItemsNil(t *testing.T) {
	// Consider nil array as not unique returns false
	var a []float64
	var x interface{} = a
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("all items in parameter %q must be unique; got:%v", c.Target, x)), true)
}

func TestValidateArrayMap_Array_UniqueItemsInvalidType(t *testing.T) {
	var x interface{} = "hello"
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("type must be array, slice or map for constraint %v; got: %v", c.Name, reflect.ValueOf(x).Kind())), true)
}

func TestValidateArrayMap_Array_UniqueItemsInvalidConstraint(t *testing.T) {
	var x interface{} = "hello"
	c := Constraint{
		Target: "str",
		Name:   "sdad",
		Rule:   true,
		Chain:  nil,
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("constraint %v is not applicable to array, slice and map type", c.Name)), true)
}

func TestValidateArrayMap_ValidateChainConstraint1(t *testing.T) {
	a := []int{1, 2, 3, 4}
	var x interface{} = a
	c := Constraint{
		Target: "str",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{"str", MaxItems, 3, nil},
		},
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("maximum item limit is %v; got: %v", (c.Chain)[0].Rule, len(a))), true)
}

func TestValidateArrayMap_ValidateChainConstraint2(t *testing.T) {
	a := []int{1, 2, 3, 4}
	var x interface{} = a
	c := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   true,
		Chain: []Constraint{
			{"str", MaxItems, 3, nil},
		},
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("maximum item limit is %v; got: %v", (c.Chain)[0].Rule, len(a))), true)
}

func TestValidateArrayMap_ValidateChainConstraint3(t *testing.T) {
	var a []string
	var x interface{} = a
	c := Constraint{
		Target: "str",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{"str", MaxItems, 3, nil},
		},
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("value can not be null; required parameter")), true)
}

func TestValidateArrayMap_ValidateChainConstraint4(t *testing.T) {
	var x interface{} = []int{}
	c := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   true,
		Chain: []Constraint{
			{"str", MaxItems, 3, nil},
		},
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("value can not be null or empty; required parameter")), true)
}

func TestValidateArrayMap_ValidateChainConstraintNilNotRequired(t *testing.T) {
	var a []int
	var x interface{} = a
	c := Constraint{
		Target: "str",
		Name:   Null,
		Rule:   false,
		Chain: []Constraint{
			{"str", MaxItems, 3, nil},
		},
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_ValidateChainConstraintEmptyNotRequired(t *testing.T) {
	var x interface{} = map[string]int{}
	c := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   false,
		Chain: []Constraint{
			{"str", MaxItems, 3, nil},
		},
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateArrayMap_ReadOnlyWithError(t *testing.T) {
	var x interface{} = []int{1, 2}
	c := Constraint{
		Target: "str",
		Name:   ReadOnly,
		Rule:   true,
		Chain: []Constraint{
			{"str", MaxItems, 3, nil},
		},
	}
	z := validateArrayMap(reflect.ValueOf(x), c)
	require.Equal(t, strings.Contains(z.Error(),
		fmt.Sprintf("readonly parameter; must send as nil or empty in request")), true)
}

func TestValidateArrayMap_ReadOnlyWithoutError(t *testing.T) {
	var x interface{} = []int{}
	c := Constraint{
		Target: "str",
		Name:   ReadOnly,
		Rule:   true,
		Chain:  nil,
	}
	require.Nil(t, validateArrayMap(reflect.ValueOf(x), c))
}

func TestValidateString_ReadOnly(t *testing.T) {
	var x interface{} = "Hello Gopher"
	c := Constraint{
		Target: "str",
		Name:   ReadOnly,
		Rule:   true,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("readonly parameter; must send as nil or empty in request")), true)
}

func TestValidateString_EmptyTrue(t *testing.T) {
	// Empty true means parameter is required but Empty returns error
	c := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   true,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(""), c).Error(),
		fmt.Sprintf("value can not be null or empty; required parameter")), true)
}

func TestValidateString_EmptyFalse(t *testing.T) {
	// Empty false means parameter is not required and Empty return nil
	var x interface{}
	c := Constraint{
		Target: "str",
		Name:   Empty,
		Rule:   false,
		Chain:  nil,
	}
	require.Nil(t, validateString(reflect.ValueOf(x), c))
}

func TestValidateString_MaxLengthInvalid(t *testing.T) {
	// Empty true means parameter is required but Empty returns error
	var x interface{} = "Hello"
	c := Constraint{
		Target: "str",
		Name:   MaxLength,
		Rule:   4,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value length must be less than or equal to %v", c.Rule)), true)
}

func TestValidateString_MaxLengthValid(t *testing.T) {
	// Empty false means parameter is not required and Empty return nil
	c := Constraint{
		Target: "str",
		Name:   MaxLength,
		Rule:   7,
		Chain:  nil,
	}
	require.Nil(t, validateString(reflect.ValueOf("Hello"), c))
}

func TestValidateString_MaxLengthRuleInvalid(t *testing.T) {
	var x interface{} = "Hello"
	c := Constraint{
		Target: "str",
		Name:   MaxLength,
		Rule:   true, // must be int for maxLength
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("rule must be integer value for %v constraint; got: %v", c.Name, c.Rule)), true)
}

func TestValidateString_MinLengthInvalid(t *testing.T) {
	var x interface{} = "Hello"
	c := Constraint{
		Target: "str",
		Name:   MinLength,
		Rule:   10,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value length must be greater than or equal to %v", c.Rule)), true)
}

func TestValidateString_MinLengthValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   MinLength,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateString(reflect.ValueOf("Hello"), c))
}

func TestValidateString_MinLengthRuleInvalid(t *testing.T) {
	var x interface{} = "Hello"
	c := Constraint{
		Target: "str",
		Name:   MinLength,
		Rule:   true, // must be int for minLength
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("rule must be integer value for %v constraint; got: %v", c.Name, c.Rule)), true)
}

func TestValidateString_PatternInvalidPattern(t *testing.T) {
	var x interface{} = "Hello"
	c := Constraint{
		Target: "str",
		Name:   Pattern,
		Rule:   `^[[:alnum:$`,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		"error parsing regexp: missing closing ]"), true)
}

func TestValidateString_PatternMatch1(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   Pattern,
		Rule:   `^http://\w+$`,
		Chain:  nil,
	}
	require.Nil(t, validateString(reflect.ValueOf("http://masd"), c))
}

func TestValidateString_PatternMatch2(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   Pattern,
		Rule:   `^[a-zA-Z0-9]+$`,
		Chain:  nil,
	}
	require.Nil(t, validateString(reflect.ValueOf("asdadad2323sad"), c))
}

func TestValidateString_PatternNotMatch(t *testing.T) {
	var x interface{} = "asdad@@ad2323sad"
	c := Constraint{
		Target: "str",
		Name:   Pattern,
		Rule:   `^[a-zA-Z0-9]+$`,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value doesn't match pattern %v", c.Rule)), true)
}

func TestValidateString_InvalidConstraint(t *testing.T) {
	var x interface{} = "asdad@@ad2323sad"
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   "^[a-zA-Z0-9]+$",
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateString(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("constraint %s is not applicable to string type", c.Name)), true)
}

func TestValidateFloat_InvalidConstraint(t *testing.T) {
	var x interface{} = 1.4
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   3.0,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("constraint %v is not applicable for type float", c.Name)), true)
}

func TestValidateFloat_InvalidRuleValue(t *testing.T) {
	var x interface{} = 1.4
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   3,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("rule must be float value for %v constraint; got: %v", c.Name, c.Rule)), true)
}

func TestValidateFloat_ExclusiveMinimumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   1.0,
		Chain:  nil,
	}
	require.Nil(t, validateFloat(reflect.ValueOf(1.42), c))
}

func TestValidateFloat_ExclusiveMinimumConstraintInvalid(t *testing.T) {
	var x interface{} = 1.4
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   1.5,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value must be greater than %v", c.Rule)), true)
}

func TestValidateFloat_ExclusiveMinimumConstraintBoundary(t *testing.T) {
	var x interface{} = 1.42
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   1.42,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value must be greater than %v", c.Rule)), true)
}

func TestValidateFloat_exclusiveMaximumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMaximum,
		Rule:   2.0,
		Chain:  nil,
	}
	require.Nil(t, validateFloat(reflect.ValueOf(1.42), c))
}

func TestValidateFloat_exclusiveMaximumConstraintInvalid(t *testing.T) {
	var x interface{} = 1.42
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMaximum,
		Rule:   1.2,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value must be less than %v", c.Rule)), true)
}

func TestValidateFloat_exclusiveMaximumConstraintBoundary(t *testing.T) {
	var x interface{} = 1.42
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMaximum,
		Rule:   1.42,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value must be less than %v", c.Rule)), true)
}

func TestValidateFloat_inclusiveMaximumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMaximum,
		Rule:   2.0,
		Chain:  nil,
	}
	require.Nil(t, validateFloat(reflect.ValueOf(1.42), c))
}

func TestValidateFloat_inclusiveMaximumConstraintInvalid(t *testing.T) {
	var x interface{} = 1.42
	c := Constraint{
		Target: "str",
		Name:   InclusiveMaximum,
		Rule:   1.2,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value must be less than or equal to %v", c.Rule)), true)

}

func TestValidateFloat_inclusiveMaximumConstraintBoundary(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMaximum,
		Rule:   1.42,
		Chain:  nil,
	}
	require.Nil(t, validateFloat(reflect.ValueOf(1.42), c))
}

func TestValidateFloat_InclusiveMinimumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMinimum,
		Rule:   1.0,
		Chain:  nil,
	}
	require.Nil(t, validateFloat(reflect.ValueOf(1.42), c))
}

func TestValidateFloat_InclusiveMinimumConstraintInvalid(t *testing.T) {
	var x interface{} = 1.42
	c := Constraint{
		Target: "str",
		Name:   InclusiveMinimum,
		Rule:   1.5,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateFloat(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("value must be greater than or equal to %v", c.Rule)), true)

}

func TestValidateFloat_InclusiveMinimumConstraintBoundary(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMinimum,
		Rule:   1.42,
		Chain:  nil,
	}
	require.Nil(t, validateFloat(reflect.ValueOf(1.42), c))
}

func TestValidateInt_InvalidConstraint(t *testing.T) {
	var x interface{} = 1
	c := Constraint{
		Target: "str",
		Name:   UniqueItems,
		Rule:   3,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateInt(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("constraint %s is not applicable for type integer", c.Name)), true)
}

func TestValidateInt_InvalidRuleValue(t *testing.T) {
	var x interface{} = 1
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   3.4,
		Chain:  nil,
	}
	require.Equal(t, strings.Contains(validateInt(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("rule must be integer value for %v constraint; got: %v", c.Name, c.Rule)), true)
}

func TestValidateInt_ExclusiveMinimumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   1,
		Chain:  nil,
	}
	require.Nil(t, validateInt(reflect.ValueOf(3), c))
}

func TestValidateInt_ExclusiveMinimumConstraintInvalid(t *testing.T) {
	var x interface{} = 1
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   3,
		Chain:  nil,
	}
	require.Equal(t, validateInt(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c, fmt.Sprintf("value must be greater than %v", c.Rule)).Error())
}

func TestValidateInt_ExclusiveMinimumConstraintBoundary(t *testing.T) {
	var x interface{} = 1
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMinimum,
		Rule:   1,
		Chain:  nil,
	}
	require.Equal(t, validateInt(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c, fmt.Sprintf("value must be greater than %v", c.Rule)).Error())
}

func TestValidateInt_exclusiveMaximumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMaximum,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateInt(reflect.ValueOf(1), c))
}

func TestValidateInt_exclusiveMaximumConstraintInvalid(t *testing.T) {
	var x interface{} = 2
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMaximum,
		Rule:   1,
		Chain:  nil,
	}
	require.Equal(t, validateInt(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c, fmt.Sprintf("value must be less than %v", c.Rule)).Error())
}

func TestValidateInt_exclusiveMaximumConstraintBoundary(t *testing.T) {
	var x interface{} = 1
	c := Constraint{
		Target: "str",
		Name:   ExclusiveMaximum,
		Rule:   1,
		Chain:  nil,
	}
	require.Equal(t, validateInt(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c, fmt.Sprintf("value must be less than %v", c.Rule)).Error())
}

func TestValidateInt_inclusiveMaximumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMaximum,
		Rule:   2,
		Chain:  nil,
	}
	require.Nil(t, validateInt(reflect.ValueOf(1), c))
}

func TestValidateInt_inclusiveMaximumConstraintInvalid(t *testing.T) {
	var x interface{} = 2
	c := Constraint{
		Target: "str",
		Name:   InclusiveMaximum,
		Rule:   1,
		Chain:  nil,
	}
	require.Equal(t, validateInt(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c, fmt.Sprintf("value must be less than or equal to %v", c.Rule)).Error())
}

func TestValidateInt_inclusiveMaximumConstraintBoundary(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMaximum,
		Rule:   1,
		Chain:  nil,
	}
	require.Nil(t, validateInt(reflect.ValueOf(1), c))
}

func TestValidateInt_InclusiveMinimumConstraintValid(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMinimum,
		Rule:   1,
		Chain:  nil,
	}
	require.Nil(t, validateInt(reflect.ValueOf(1), c))
}

func TestValidateInt_InclusiveMinimumConstraintInvalid(t *testing.T) {
	var x interface{} = 1
	c := Constraint{
		Target: "str",
		Name:   InclusiveMinimum,
		Rule:   2,
		Chain:  nil,
	}
	require.Equal(t, validateInt(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c, fmt.Sprintf("value must be greater than or equal to %v", c.Rule)).Error())
}

func TestValidateInt_InclusiveMinimumConstraintBoundary(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   InclusiveMinimum,
		Rule:   1,
		Chain:  nil,
	}
	require.Nil(t, validateInt(reflect.ValueOf(1), c))
}

func TestValidateInt_MultipleOfWithoutError(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   MultipleOf,
		Rule:   10,
		Chain:  nil,
	}
	require.Nil(t, validateInt(reflect.ValueOf(2300), c))
}

func TestValidateInt_MultipleOfWithError(t *testing.T) {
	c := Constraint{
		Target: "str",
		Name:   MultipleOf,
		Rule:   11,
		Chain:  nil,
	}
	require.Equal(t, validateInt(reflect.ValueOf(2300), c).Error(),
		createError(reflect.ValueOf(2300), c, fmt.Sprintf("value must be a multiple of %v", c.Rule)).Error())
}

func TestValidatePointer_NilTrue(t *testing.T) {
	var z *int
	var x interface{} = z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true, // Required property
		Chain:  nil,
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c, "value can not be null; required parameter").Error())
}

func TestValidatePointer_NilFalse(t *testing.T) {
	var z *int
	var x interface{} = z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   false, // not required property
		Chain:  nil,
	}
	require.Nil(t, validatePtr(reflect.ValueOf(x), c))
}

func TestValidatePointer_NilReadonlyValid(t *testing.T) {
	var z *int
	var x interface{} = z
	c := Constraint{
		Target: "ptr",
		Name:   ReadOnly,
		Rule:   true,
		Chain:  nil,
	}
	require.Nil(t, validatePtr(reflect.ValueOf(x), c))
}

func TestValidatePointer_NilReadonlyInvalid(t *testing.T) {
	z := 10
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   ReadOnly,
		Rule:   true,
		Chain:  nil,
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(z), c, "readonly parameter; must send as nil or empty in request").Error())
}

func TestValidatePointer_IntValid(t *testing.T) {
	z := 10
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   InclusiveMinimum,
		Rule:   3,
		Chain:  nil,
	}
	require.Nil(t, validatePtr(reflect.ValueOf(x), c))
}

func TestValidatePointer_IntInvalid(t *testing.T) {
	z := 10
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   InclusiveMinimum,
				Rule:   11,
				Chain:  nil,
			},
		},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(10), c.Chain[0], "value must be greater than or equal to 11").Error())
}

func TestValidatePointer_IntInvalidConstraint(t *testing.T) {
	z := 10
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   MaxItems,
				Rule:   3,
				Chain:  nil,
			},
		},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(10), c.Chain[0],
			fmt.Sprintf("constraint %v is not applicable for type integer", MaxItems)).Error())
}

func TestValidatePointer_ValidInt64(t *testing.T) {
	z := int64(10)
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   InclusiveMinimum,
				Rule:   3,
				Chain:  nil,
			},
		}}
	require.Nil(t, validatePtr(reflect.ValueOf(x), c))
}

func TestValidatePointer_InvalidConstraintInt64(t *testing.T) {
	z := int64(10)
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   MaxItems,
				Rule:   3,
				Chain:  nil,
			},
		},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(10), c.Chain[0],
			fmt.Sprintf("constraint %v is not applicable for type integer", MaxItems)).Error())
}

func TestValidatePointer_ValidFloat(t *testing.T) {
	z := 10.1
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   InclusiveMinimum,
				Rule:   3.0,
				Chain:  nil,
			}}}
	require.Nil(t, validatePtr(reflect.ValueOf(x), c))
}

func TestValidatePointer_InvalidFloat(t *testing.T) {
	z := 10.1
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   InclusiveMinimum,
				Rule:   12.0,
				Chain:  nil,
			}},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(10.1), c.Chain[0],
			"value must be greater than or equal to 12").Error())
}

func TestValidatePointer_InvalidConstraintFloat(t *testing.T) {
	z := 10.1
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   MaxItems,
				Rule:   3.0,
				Chain:  nil,
			}},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(10.1), c.Chain[0],
			fmt.Sprintf("constraint %v is not applicable for type float", MaxItems)).Error())
}

func TestValidatePointer_StringValid(t *testing.T) {
	z := "hello"
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   Pattern,
				Rule:   "^[a-z]+$",
				Chain:  nil,
			}}}
	require.Nil(t, validatePtr(reflect.ValueOf(x), c))
}

func TestValidatePointer_StringInvalid(t *testing.T) {
	z := "hello"
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   MaxLength,
				Rule:   2,
				Chain:  nil,
			}}}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf("hello"), c.Chain[0],
			"value length must be less than or equal to 2").Error())
}

func TestValidatePointer_ArrayValid(t *testing.T) {
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   UniqueItems,
				Rule:   "true",
				Chain:  nil,
			}}}
	require.Nil(t, validatePtr(reflect.ValueOf(&[]string{"1", "2"}), c))
}

func TestValidatePointer_ArrayInvalid(t *testing.T) {
	z := []string{"1", "2", "2"}
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{{
			Target: "ptr",
			Name:   UniqueItems,
			Rule:   true,
			Chain:  nil,
		}},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(z), c.Chain[0],
			fmt.Sprintf("all items in parameter %q must be unique; got:%v", c.Target, z)).Error())
}

func TestValidatePointer_MapValid(t *testing.T) {
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{
			{
				Target: "ptr",
				Name:   UniqueItems,
				Rule:   true,
				Chain:  nil,
			}}}
	require.Nil(t, validatePtr(reflect.ValueOf(&map[interface{}]string{1: "1", "1": "2"}), c))
}

func TestValidatePointer_MapInvalid(t *testing.T) {
	z := map[interface{}]string{1: "1", "1": "2", 1.3: "2"}
	var x interface{} = &z
	c := Constraint{
		Target: "ptr",
		Name:   Null,
		Rule:   true,
		Chain: []Constraint{{
			Target: "ptr",
			Name:   UniqueItems,
			Rule:   true,
			Chain:  nil,
		}},
	}
	require.Equal(t, strings.Contains(validatePtr(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("all items in parameter %q must be unique;", c.Target)), true)
}

type Child struct {
	I string
}
type Product struct {
	C    *Child
	Str  *string
	Name string
	Arr  *[]string
	M    *map[string]string
	Num  *int32
}

type Sample struct {
	M    *map[string]*string
	Name string
}

func TestValidatePointer_StructWithError(t *testing.T) {
	s := "hello"
	var x interface{} = &Product{
		C:    &Child{"100"},
		Str:  &s,
		Name: "Gopher",
	}
	c := Constraint{
		"p", Null, "True",
		[]Constraint{
			{"C", Null, true,
				[]Constraint{
					{"I", MaxLength, 2, nil},
				}},
			{"Str", MaxLength, 2, nil},
			{"Name", MaxLength, 5, nil},
		},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf("100"), c.Chain[0].Chain[0],
			"value length must be less than or equal to 2").Error())
}

func TestValidatePointer_WithNilStruct(t *testing.T) {
	var p *Product
	var x interface{} = p
	c := Constraint{
		"p", Null, true,
		[]Constraint{
			{"C", Null, true,
				[]Constraint{
					{"I", Empty, true,
						[]Constraint{
							{"I", MaxLength, 5, nil},
						}},
				}},
			{"Str", MaxLength, 2, nil},
			{"Name", MaxLength, 5, nil},
		},
	}
	require.Equal(t, validatePtr(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x), c,
			fmt.Sprintf("value can not be null; required parameter")).Error())
}

func TestValidatePointer_StructWithNoError(t *testing.T) {
	s := "hello"
	var x interface{} = &Product{
		C:    &Child{"100"},
		Str:  &s,
		Name: "Gopher",
	}
	c := Constraint{
		"p", Null, true,
		[]Constraint{
			{"C", Null, true,
				[]Constraint{
					{"I", Empty, true,
						[]Constraint{
							{"I", MaxLength, 5, nil},
						}},
				}},
		},
	}
	require.Nil(t, validatePtr(reflect.ValueOf(x), c))
}

func TestValidateStruct_FieldNotExist(t *testing.T) {
	s := "hello"
	var x interface{} = Product{
		C:    &Child{"100"},
		Str:  &s,
		Name: "Gopher",
	}
	c := Constraint{
		"C", Null, true,
		[]Constraint{
			{"Name", Empty, true, nil},
		},
	}
	s = "Name"
	require.Equal(t, validateStruct(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(Child{"100"}), c.Chain[0],
			fmt.Sprintf("field %q doesn't exist", s)).Error())
}

func TestValidateStruct_WithChainConstraint(t *testing.T) {
	s := "hello"
	var x interface{} = Product{
		C:    &Child{"100"},
		Str:  &s,
		Name: "Gopher",
	}
	c := Constraint{
		"C", Null, true,
		[]Constraint{
			{"I", Empty, true,
				[]Constraint{
					{"I", MaxLength, 2, nil},
				}},
		},
	}
	require.Equal(t, validateStruct(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf("100"), c.Chain[0].Chain[0], "value length must be less than or equal to 2").Error())
}

func TestValidateStruct_WithoutChainConstraint(t *testing.T) {
	s := "hello"
	var x interface{} = Product{
		C:    &Child{""},
		Str:  &s,
		Name: "Gopher",
	}
	c := Constraint{"C", Null, true,
		[]Constraint{
			{"I", Empty, true, nil}, // throw error for Empty
		}}
	require.Equal(t, validateStruct(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(""), c.Chain[0], "value can not be null or empty; required parameter").Error())
}

func TestValidateStruct_WithArrayNull(t *testing.T) {
	s := "hello"
	var x interface{} = Product{
		C:    &Child{""},
		Str:  &s,
		Name: "Gopher",
		Arr:  nil,
	}
	c := Constraint{"Arr", Null, true,
		[]Constraint{
			{"Arr", MaxItems, 4, nil},
			{"Arr", MinItems, 2, nil},
		},
	}
	require.Equal(t, validateStruct(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(x.(Product).Arr), c, "value can not be null; required parameter").Error())
}

func TestValidateStruct_WithArrayEmptyError(t *testing.T) {
	// arr := []string{}
	var x interface{} = Product{
		Arr: &[]string{},
	}
	c := Constraint{
		"Arr", Null, true,
		[]Constraint{
			{"Arr", Empty, true, nil},
			{"Arr", MaxItems, 4, nil},
			{"Arr", MinItems, 2, nil},
		}}

	require.Equal(t, validateStruct(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(*(x.(Product).Arr)), c.Chain[0],
			fmt.Sprintf("value can not be null or empty; required parameter")).Error())
}

func TestValidateStruct_WithArrayEmptyWithoutError(t *testing.T) {
	var x interface{} = Product{
		Arr: &[]string{},
	}
	c := Constraint{
		"Arr", Null, true,
		[]Constraint{
			{"Arr", Empty, false, nil},
			{"Arr", MaxItems, 4, nil},
		},
	}
	require.Nil(t, validateStruct(reflect.ValueOf(x), c))
}

func TestValidateStruct_ArrayWithError(t *testing.T) {
	arr := []string{"1", "1"}
	var x interface{} = Product{
		Arr: &arr,
	}
	c := Constraint{
		"Arr", Null, true,
		[]Constraint{
			{"Arr", Empty, true, nil},
			{"Arr", MaxItems, 4, nil},
			{"Arr", UniqueItems, true, nil},
		},
	}
	s := "Arr"
	require.Equal(t, validateStruct(reflect.ValueOf(x), c).Error(),
		createError(reflect.ValueOf(*(x.(Product).Arr)), c.Chain[2],
			fmt.Sprintf("all items in parameter %q must be unique; got:%v", s, *(x.(Product).Arr))).Error())
}

func TestValidateStruct_MapWithError(t *testing.T) {
	m := map[string]string{
		"a": "hello",
		"b": "hello",
	}
	var x interface{} = Product{
		M: &m,
	}
	c := Constraint{
		"M", Null, true,
		[]Constraint{
			{"M", Empty, true, nil},
			{"M", MaxItems, 4, nil},
			{"M", UniqueItems, true, nil},
		},
	}

	s := "M"
	require.Equal(t, strings.Contains(validateStruct(reflect.ValueOf(x), c).Error(),
		fmt.Sprintf("all items in parameter %q must be unique;", s)), true)
}

func TestValidateStruct_MapWithNoError(t *testing.T) {
	m := map[string]string{}
	var x interface{} = Product{
		M: &m,
	}
	c := Constraint{
		"M", Null, true,
		[]Constraint{
			{"M", Empty, false, nil},
			{"M", MaxItems, 4, nil},
		},
	}
	require.Nil(t, validateStruct(reflect.ValueOf(x), c))
}

func TestValidateStruct_MapNilNoError(t *testing.T) {
	var m map[string]string
	var x interface{} = Product{
		M: &m,
	}
	c := Constraint{
		"M", Null, false,
		[]Constraint{
			{"M", Empty, false, nil},
			{"M", MaxItems, 4, nil},
		},
	}
	require.Nil(t, validateStruct(reflect.ValueOf(x), c))
}

func TestValidate_MapValidationWithError(t *testing.T) {
	var x1 interface{} = &Product{
		Arr: &[]string{"1", "2"},
		M:   &map[string]string{"a": "hello"},
	}
	s := "hello"
	var x2 interface{} = &Sample{
		M: &map[string]*string{"a": &s},
	}
	v := []Validation{
		{x1,
			[]Constraint{{"x1", Null, true,
				[]Constraint{
					{"Arr", Null, true,
						[]Constraint{
							{"Arr", Empty, true, nil},
							{"Arr", MaxItems, 4, nil},
							{"Arr", UniqueItems, true, nil},
						},
					},
					{"M", Null, false,
						[]Constraint{
							{"M", Empty, false, nil},
							{"M", MinItems, 1, nil},
							{"M", UniqueItems, true, nil},
						},
					},
				},
			}}},
		{x2,
			[]Constraint{
				{"x2", Null, true,
					[]Constraint{
						{"M", Null, false,
							[]Constraint{
								{"M", Empty, false, nil},
								{"M", MinItems, 2, nil},
								{"M", UniqueItems, true, nil},
							},
						},
					},
				},
				{"Name", Empty, true, nil},
			}},
	}

	z := Validate(v).Error()
	require.Equal(t, strings.Contains(z, "minimum item limit is 2; got: 1"), true)
	require.Equal(t, strings.Contains(z, "MinItems"), true)
}

func TestValidate_MapValidationWithoutError(t *testing.T) {
	var x1 interface{} = &Product{
		Arr: &[]string{"1", "2"},
		M:   &map[string]string{"a": "hello"},
	}
	s := "hello"
	var x2 interface{} = &Sample{
		M: &map[string]*string{"a": &s},
	}
	v := []Validation{
		{x1,
			[]Constraint{{"x1", Null, true,
				[]Constraint{
					{"Arr", Null, true,
						[]Constraint{
							{"Arr", Empty, true, nil},
							{"Arr", MaxItems, 4, nil},
							{"Arr", UniqueItems, true, nil},
						},
					},
					{"M", Null, false,
						[]Constraint{
							{"M", Empty, false, nil},
							{"M", MinItems, 1, nil},
							{"M", UniqueItems, true, nil},
							{"M", Pattern, "^[a-z]+$", nil},
						},
					},
				},
			}}},
		{x2,
			[]Constraint{
				{"x2", Null, true,
					[]Constraint{
						{"M", Null, false,
							[]Constraint{
								{"M", Empty, false, nil},
								{"M", MinItems, 1, nil},
								{"M", UniqueItems, true, nil},
								{"M", Pattern, "^[a-z]+$", nil},
							},
						},
					},
				},
				{"Name", Empty, true, nil},
			}},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_UnknownType(t *testing.T) {
	var c chan int
	v := []Validation{
		{c,
			[]Constraint{{"c", Null, true, nil}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(c), v[0].Constraints[0],
			fmt.Sprintf("unknown type %v", reflect.ValueOf(c).Kind())).Error())
}

func TestValidate_example1(t *testing.T) {
	var x1 interface{} = Product{
		Arr: &[]string{"1", "1"},
		M:   &map[string]string{"a": "hello"},
	}
	s := "hello"
	var x2 interface{} = Sample{
		M: &map[string]*string{"a": &s},
	}
	v := []Validation{
		{x1,
			[]Constraint{{"Arr", Null, true,
				[]Constraint{
					{"Arr", Empty, true, nil},
					{"Arr", MaxItems, 4, nil},
					{"Arr", UniqueItems, true, nil},
				}},
				{"M", Null, false,
					[]Constraint{
						{"M", Empty, false, nil},
						{"M", MinItems, 1, nil},
						{"M", UniqueItems, true, nil},
					},
				},
			}},
		{x2,
			[]Constraint{
				{"M", Null, false,
					[]Constraint{
						{"M", Empty, false, nil},
						{"M", MinItems, 1, nil},
						{"M", UniqueItems, true, nil},
					},
				},
				{"Name", Empty, true, nil},
			}},
	}
	s = "Arr"
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf([]string{"1", "1"}), v[0].Constraints[0].Chain[2],
			fmt.Sprintf("all items in parameter %q must be unique; got:%v", s, []string{"1", "1"})).Error())
}

func TestValidate_Int(t *testing.T) {
	n := int32(100)
	v := []Validation{
		{n,
			[]Constraint{
				{"n", MultipleOf, 10, nil},
				{"n", ExclusiveMinimum, 100, nil},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(n), v[0].Constraints[1],
			"value must be greater than 100").Error())
}

func TestValidate_IntPointer(t *testing.T) {
	n := int32(100)
	p := &n
	v := []Validation{
		{p,
			[]Constraint{
				{"p", Null, true, []Constraint{
					{"p", ExclusiveMinimum, 100, nil},
				}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(n), v[0].Constraints[0].Chain[0],
			"value must be greater than 100").Error())

	// required paramter
	p = nil
	v = []Validation{
		{p,
			[]Constraint{
				{"p", Null, true, []Constraint{
					{"p", ExclusiveMinimum, 100, nil},
				}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(v[0].TargetValue), v[0].Constraints[0],
			"value can not be null; required parameter").Error())

	// Not required
	p = nil
	v = []Validation{
		{p,
			[]Constraint{
				{"p", Null, false, []Constraint{
					{"p", ExclusiveMinimum, 100, nil},
				}},
			},
		},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_IntStruct(t *testing.T) {
	n := int32(100)
	p := &Product{
		Num: &n,
	}

	v := []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"Num", Null, true, []Constraint{
					{"Num", ExclusiveMinimum, 100, nil},
				}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(n), v[0].Constraints[0].Chain[0].Chain[0],
			"value must be greater than 100").Error())

	// required paramter
	p = &Product{}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"p.Num", Null, true, []Constraint{
					{"p.Num", ExclusiveMinimum, 100, nil},
				}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(p.Num), v[0].Constraints[0].Chain[0],
			"value can not be null; required parameter").Error())

	// Not required
	p = &Product{}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"Num", Null, false, []Constraint{
					{"Num", ExclusiveMinimum, 100, nil},
				}},
			},
		}}},
	}
	require.Nil(t, Validate(v))

	// Parent not required
	p = nil
	v = []Validation{
		{p, []Constraint{{"p", Null, false,
			[]Constraint{
				{"Num", Null, false, []Constraint{
					{"Num", ExclusiveMinimum, 100, nil},
				}},
			},
		}}},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_String(t *testing.T) {
	s := "hello"
	v := []Validation{
		{s,
			[]Constraint{
				{"s", Empty, true, nil},
				{"s", Empty, true,
					[]Constraint{{"s", MaxLength, 3, nil}}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s), v[0].Constraints[1].Chain[0],
			"value length must be less than or equal to 3").Error())

	// required paramter
	s = ""
	v = []Validation{
		{s,
			[]Constraint{
				{"s", Empty, true, nil},
				{"s", Empty, true,
					[]Constraint{{"s", MaxLength, 3, nil}}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s), v[0].Constraints[1],
			"value can not be null or empty; required parameter").Error())

	// not required paramter
	s = ""
	v = []Validation{
		{s,
			[]Constraint{
				{"s", Empty, false, nil},
				{"s", Empty, false,
					[]Constraint{{"s", MaxLength, 3, nil}}},
			},
		},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_StringStruct(t *testing.T) {
	s := "hello"
	p := &Product{
		Str: &s,
	}

	v := []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"p.Str", Null, true, []Constraint{
					{"p.Str", Empty, true, nil},
					{"p.Str", MaxLength, 3, nil},
				}},
			},
		}}},
	}
	// e := ValidationError{
	// 	Constraint:  MaxLength,
	// 	Target:      "Str",
	// 	TargetValue: s,
	// 	Details:     fmt.Sprintf("value length must be less than 3", s),
	// }
	// if z := Validate(v); !reflect.DeepEqual(e, z) {
	// 	t.Fatalf("autorest/validation: Validate failed to return error \nexpect: %v\ngot: %v", e, z)
	// }
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s), v[0].Constraints[0].Chain[0].Chain[1],
			"value length must be less than or equal to 3").Error())

	// required paramter - can't be Empty
	s = ""
	p = &Product{
		Str: &s,
	}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"Str", Null, true, []Constraint{
					{"Str", Empty, true, nil},
					{"Str", MaxLength, 3, nil},
				}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s), v[0].Constraints[0].Chain[0].Chain[0],
			"value can not be null or empty; required parameter").Error())

	// required paramter - can't be null
	p = &Product{}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"p.Str", Null, true, []Constraint{
					{"p.Str", Empty, true, nil},
					{"p.Str", MaxLength, 3, nil},
				}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(p.Str), v[0].Constraints[0].Chain[0],
			"value can not be null; required parameter").Error())

	// Not required
	p = &Product{}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"Str", Null, false, []Constraint{
					{"Str", Empty, true, nil},
					{"Str", MaxLength, 3, nil},
				}},
			},
		}}},
	}
	require.Nil(t, Validate(v))

	// Parent not required
	p = nil
	v = []Validation{
		{p, []Constraint{{"p", Null, false,
			[]Constraint{
				{"Str", Null, true, []Constraint{
					{"Str", Empty, true, nil},
					{"Str", MaxLength, 3, nil},
				}},
			},
		}}},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_Array(t *testing.T) {
	s := []string{"hello"}
	v := []Validation{
		{s,
			[]Constraint{
				{"s", Null, true,
					[]Constraint{
						{"s", Empty, true, nil},
						{"s", MinItems, 2, nil},
					}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s), v[0].Constraints[0].Chain[1],
			fmt.Sprintf("minimum item limit is 2; got: %v", len(s))).Error())

	// Empty array
	v = []Validation{
		{[]string{},
			[]Constraint{
				{"s", Null, true,
					[]Constraint{
						{"s", Empty, true, nil},
						{"s", MinItems, 2, nil}}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf([]string{}), v[0].Constraints[0].Chain[0],
			"value can not be null or empty; required parameter").Error())

	// null array
	var s1 []string
	v = []Validation{
		{s1,
			[]Constraint{
				{"s1", Null, true,
					[]Constraint{
						{"s1", Empty, true, nil},
						{"s1", MinItems, 2, nil}}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s1), v[0].Constraints[0],
			"value can not be null; required parameter").Error())

	// not required paramter
	v = []Validation{
		{s1,
			[]Constraint{
				{"s1", Null, false,
					[]Constraint{
						{"s1", Empty, true, nil},
						{"s1", MinItems, 2, nil}}},
			},
		},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_ArrayPointer(t *testing.T) {
	s := []string{"hello"}
	v := []Validation{
		{&s,
			[]Constraint{
				{"s", Null, true,
					[]Constraint{
						{"s", Empty, true, nil},
						{"s", MinItems, 2, nil},
					}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s), v[0].Constraints[0].Chain[1],
			fmt.Sprintf("minimum item limit is 2; got: %v", len(s))).Error())

	// Empty array
	v = []Validation{
		{&[]string{},
			[]Constraint{
				{"s", Null, true,
					[]Constraint{
						{"s", Empty, true, nil},
						{"s", MinItems, 2, nil}}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf([]string{}), v[0].Constraints[0].Chain[0],
			"value can not be null or empty; required parameter").Error())

	// null array
	var s1 *[]string
	v = []Validation{
		{s1,
			[]Constraint{
				{"s1", Null, true,
					[]Constraint{
						{"s1", Empty, true, nil},
						{"s1", MinItems, 2, nil}}},
			},
		},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s1), v[0].Constraints[0],
			"value can not be null; required parameter").Error())

	// not required paramter
	v = []Validation{
		{s1,
			[]Constraint{
				{"s1", Null, false,
					[]Constraint{
						{"s1", Empty, true, nil},
						{"s1", MinItems, 2, nil}}},
			},
		},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_ArrayInStruct(t *testing.T) {
	s := []string{"hello"}
	p := &Product{
		Arr: &s,
	}

	v := []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"p.Arr", Null, true, []Constraint{
					{"p.Arr", Empty, true, nil},
					{"p.Arr", MinItems, 2, nil},
				}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(s), v[0].Constraints[0].Chain[0].Chain[1],
			fmt.Sprintf("minimum item limit is 2; got: %v", len(s))).Error())

	// required paramter - can't be Empty
	p = &Product{
		Arr: &[]string{},
	}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"p.Arr", Null, true, []Constraint{
					{"p.Arr", Empty, true, nil},
					{"p.Arr", MinItems, 2, nil},
				}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf([]string{}), v[0].Constraints[0].Chain[0].Chain[0],
			"value can not be null or empty; required parameter").Error())

	// required paramter - can't be null
	p = &Product{}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{
				{"p.Arr", Null, true, []Constraint{
					{"p.Arr", Empty, true, nil},
					{"p.Arr", MinItems, 2, nil},
				}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(p.Arr), v[0].Constraints[0].Chain[0],
			"value can not be null; required parameter").Error())

	// Not required
	v = []Validation{
		{&Product{}, []Constraint{{"p", Null, true,
			[]Constraint{
				{"Arr", Null, false, []Constraint{
					{"Arr", Empty, true, nil},
					{"Arr", MinItems, 2, nil},
				}},
			},
		}}},
	}
	require.Nil(t, Validate(v))

	// Parent not required
	p = nil
	v = []Validation{
		{p, []Constraint{{"p", Null, false,
			[]Constraint{
				{"Arr", Null, true, []Constraint{
					{"Arr", Empty, true, nil},
					{"Arr", MinItems, 2, nil},
				}},
			},
		}}},
	}
	require.Nil(t, Validate(v))
}

func TestValidate_StructInStruct(t *testing.T) {
	p := &Product{
		C: &Child{I: "hello"},
	}
	v := []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{{"C", Null, true,
				[]Constraint{{"I", MinLength, 7, nil}}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(p.C.I), v[0].Constraints[0].Chain[0].Chain[0],
			"value length must be greater than or equal to 7").Error())

	// required paramter - can't be Empty
	p = &Product{
		C: &Child{I: ""},
	}

	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{{"C", Null, true,
				[]Constraint{{"I", Empty, true, nil}}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(p.C.I), v[0].Constraints[0].Chain[0].Chain[0],
			"value can not be null or empty; required parameter").Error())

	// required paramter - can't be null
	p = &Product{}
	v = []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{{"C", Null, true,
				[]Constraint{{"I", Empty, true, nil}}},
			},
		}}},
	}
	require.Equal(t, Validate(v).Error(),
		createError(reflect.ValueOf(p.C), v[0].Constraints[0].Chain[0],
			"value can not be null; required parameter").Error())

	// Not required
	v = []Validation{
		{&Product{}, []Constraint{{"p", Null, true,
			[]Constraint{{"p.C", Null, false,
				[]Constraint{{"p.C.I", Empty, true, nil}}},
			},
		}}},
	}
	require.Nil(t, Validate(v))

	// Parent not required
	p = nil
	v = []Validation{
		{p, []Constraint{{"p", Null, false,
			[]Constraint{{"p.C", Null, false,
				[]Constraint{{"p.C.I", Empty, true, nil}}},
			},
		}}},
	}
	require.Nil(t, Validate(v))
}

func TestNewErrorWithValidationError(t *testing.T) {
	p := &Product{}
	v := []Validation{
		{p, []Constraint{{"p", Null, true,
			[]Constraint{{"p.C", Null, true,
				[]Constraint{{"p.C.I", Empty, true, nil}}},
			},
		}}},
	}
	err := createError(reflect.ValueOf(p.C), v[0].Constraints[0].Chain[0], "value can not be null; required parameter")
	z := fmt.Sprintf("batch.AccountClient#Create: Invalid input: %s",
		err.Error())
	require.Equal(t, NewErrorWithValidationError(err, "batch.AccountClient", "Create").Error(), z)
}
