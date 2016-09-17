package lib

func AbsInt(a int) int {
	if a > 0 {
		return a
	}
	return a * -1
}

func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func MinInt(a, b int) int {
	if a > b {
		return b
	}
	return a
}
