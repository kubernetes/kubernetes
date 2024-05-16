package govalidator

import (
	"math"
	"reflect"
)

// Abs returns absolute value of number
func Abs(value float64) float64 {
	return math.Abs(value)
}

// Sign returns signum of number: 1 in case of value > 0, -1 in case of value < 0, 0 otherwise
func Sign(value float64) float64 {
	if value > 0 {
		return 1
	} else if value < 0 {
		return -1
	} else {
		return 0
	}
}

// IsNegative returns true if value < 0
func IsNegative(value float64) bool {
	return value < 0
}

// IsPositive returns true if value > 0
func IsPositive(value float64) bool {
	return value > 0
}

// IsNonNegative returns true if value >= 0
func IsNonNegative(value float64) bool {
	return value >= 0
}

// IsNonPositive returns true if value <= 0
func IsNonPositive(value float64) bool {
	return value <= 0
}

// InRange returns true if value lies between left and right border
func InRangeInt(value, left, right interface{}) bool {
	value64, _ := ToInt(value)
	left64, _ := ToInt(left)
	right64, _ := ToInt(right)
	if left64 > right64 {
		left64, right64 = right64, left64
	}
	return value64 >= left64 && value64 <= right64
}

// InRange returns true if value lies between left and right border
func InRangeFloat32(value, left, right float32) bool {
	if left > right {
		left, right = right, left
	}
	return value >= left && value <= right
}

// InRange returns true if value lies between left and right border
func InRangeFloat64(value, left, right float64) bool {
	if left > right {
		left, right = right, left
	}
	return value >= left && value <= right
}

// InRange returns true if value lies between left and right border, generic type to handle int, float32 or float64, all types must the same type
func InRange(value interface{}, left interface{}, right interface{}) bool {

	reflectValue := reflect.TypeOf(value).Kind()
	reflectLeft := reflect.TypeOf(left).Kind()
	reflectRight := reflect.TypeOf(right).Kind()

	if reflectValue == reflect.Int && reflectLeft == reflect.Int && reflectRight == reflect.Int {
		return InRangeInt(value.(int), left.(int), right.(int))
	} else if reflectValue == reflect.Float32 && reflectLeft == reflect.Float32 && reflectRight == reflect.Float32 {
		return InRangeFloat32(value.(float32), left.(float32), right.(float32))
	} else if reflectValue == reflect.Float64 && reflectLeft == reflect.Float64 && reflectRight == reflect.Float64 {
		return InRangeFloat64(value.(float64), left.(float64), right.(float64))
	} else {
		return false
	}
}

// IsWhole returns true if value is whole number
func IsWhole(value float64) bool {
	return math.Remainder(value, 1) == 0
}

// IsNatural returns true if value is natural number (positive and whole)
func IsNatural(value float64) bool {
	return IsWhole(value) && IsPositive(value)
}
