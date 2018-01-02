package main

import (
	"math/rand"
)

// chunkStrings chunks the string slice
func chunkStrings(x []string, numChunks int) [][]string {
	var result [][]string
	chunkSize := (len(x) + numChunks - 1) / numChunks
	for i := 0; i < len(x); i += chunkSize {
		ub := i + chunkSize
		if ub > len(x) {
			ub = len(x)
		}
		result = append(result, x[i:ub])
	}
	return result
}

// shuffleStrings shuffles strings
func shuffleStrings(x []string, seed int64) {
	r := rand.New(rand.NewSource(seed))
	for i := range x {
		j := r.Intn(i + 1)
		x[i], x[j] = x[j], x[i]
	}
}
