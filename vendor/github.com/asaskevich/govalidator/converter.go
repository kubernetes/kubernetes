package govalidator

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// ToString convert the input to a string.
func ToString(obj interface{}) string {
	res := fmt.Sprintf("%v", obj)
	return string(res)
}

// ToJSON convert the input to a valid JSON string
func ToJSON(obj interface{}) (string, error) {
	res, err := json.Marshal(obj)
	if err != nil {
		res = []byte("")
	}
	return string(res), err
}

// ToFloat convert the input string to a float, or 0.0 if the input is not a float.
func ToFloat(str string) (float64, error) {
	res, err := strconv.ParseFloat(str, 64)
	if err != nil {
		res = 0.0
	}
	return res, err
}

// ToInt convert the input string to an integer, or 0 if the input is not an integer.
func ToInt(str string) (int64, error) {
	res, err := strconv.ParseInt(str, 0, 64)
	if err != nil {
		res = 0
	}
	return res, err
}

// ToBoolean convert the input string to a boolean.
func ToBoolean(str string) (bool, error) {
	res, err := strconv.ParseBool(str)
	if err != nil {
		res = false
	}
	return res, err
}
