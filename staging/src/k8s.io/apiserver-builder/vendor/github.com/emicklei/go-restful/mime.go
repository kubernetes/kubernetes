package restful

import (
	"strconv"
	"strings"
)

type mime struct {
	media   string
	quality float64
}

// insertMime adds a mime to a list and keeps it sorted by quality.
func insertMime(l []mime, e mime) []mime {
	for i, each := range l {
		// if current mime has lower quality then insert before
		if e.quality > each.quality {
			left := append([]mime{}, l[0:i]...)
			return append(append(left, e), l[i:]...)
		}
	}
	return append(l, e)
}

// sortedMimes returns a list of mime sorted (desc) by its specified quality.
func sortedMimes(accept string) (sorted []mime) {
	for _, each := range strings.Split(accept, ",") {
		typeAndQuality := strings.Split(strings.Trim(each, " "), ";")
		if len(typeAndQuality) == 1 {
			sorted = insertMime(sorted, mime{typeAndQuality[0], 1.0})
		} else {
			// take factor
			parts := strings.Split(typeAndQuality[1], "=")
			if len(parts) == 2 {
				f, err := strconv.ParseFloat(parts[1], 64)
				if err != nil {
					traceLogger.Printf("unable to parse quality in %s, %v", each, err)
				} else {
					sorted = insertMime(sorted, mime{typeAndQuality[0], f})
				}
			}
		}
	}
	return
}
