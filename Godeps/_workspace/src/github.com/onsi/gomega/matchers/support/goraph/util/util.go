package util

import "math"

func Odd(n int) bool {
    return math.Mod(float64(n), 2.0) == 1.0
} 
