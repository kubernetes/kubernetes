package p

import (
	`fmt`
	f "fmt"
)

import math `math` // to do math things
import fm "fmt"
import m "math"

import (
	y "math"
	z "math"
)

func foo(unrelated int) {
	_ = f
	_ = fmt.Println
	_ = Printf
	_ = f.Println
	_ = m.Sin
	_ = fm.Print
	_ = math.Cos
	_, _, _, _ = w.Tan, x.Tan, y.Tan, z.Tan
	
	lookup := make(map[interface{}]int)
	lookup[z.Pi] = 3

	{
		_ = math.Sin
		_ = t.Sin
	}
}
