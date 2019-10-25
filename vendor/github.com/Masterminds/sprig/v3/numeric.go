package sprig

import (
	"fmt"
	"math"
	"strconv"

	"github.com/spf13/cast"
)

// toFloat64 converts 64-bit floats
func toFloat64(v interface{}) float64 {
	return cast.ToFloat64(v)
}

func toInt(v interface{}) int {
	return cast.ToInt(v)
}

// toInt64 converts integer types to 64-bit integers
func toInt64(v interface{}) int64 {
	return cast.ToInt64(v)
}

func max(a interface{}, i ...interface{}) int64 {
	aa := toInt64(a)
	for _, b := range i {
		bb := toInt64(b)
		if bb > aa {
			aa = bb
		}
	}
	return aa
}

func min(a interface{}, i ...interface{}) int64 {
	aa := toInt64(a)
	for _, b := range i {
		bb := toInt64(b)
		if bb < aa {
			aa = bb
		}
	}
	return aa
}

func until(count int) []int {
	step := 1
	if count < 0 {
		step = -1
	}
	return untilStep(0, count, step)
}

func untilStep(start, stop, step int) []int {
	v := []int{}

	if stop < start {
		if step >= 0 {
			return v
		}
		for i := start; i > stop; i += step {
			v = append(v, i)
		}
		return v
	}

	if step <= 0 {
		return v
	}
	for i := start; i < stop; i += step {
		v = append(v, i)
	}
	return v
}

func floor(a interface{}) float64 {
	aa := toFloat64(a)
	return math.Floor(aa)
}

func ceil(a interface{}) float64 {
	aa := toFloat64(a)
	return math.Ceil(aa)
}

func round(a interface{}, p int, rOpt ...float64) float64 {
	roundOn := .5
	if len(rOpt) > 0 {
		roundOn = rOpt[0]
	}
	val := toFloat64(a)
	places := toFloat64(p)

	var round float64
	pow := math.Pow(10, places)
	digit := pow * val
	_, div := math.Modf(digit)
	if div >= roundOn {
		round = math.Ceil(digit)
	} else {
		round = math.Floor(digit)
	}
	return round / pow
}

// converts unix octal to decimal
func toDecimal(v interface{}) int64 {
	result, err := strconv.ParseInt(fmt.Sprint(v), 8, 64)
	if err != nil {
		return 0
	}
	return result
}
