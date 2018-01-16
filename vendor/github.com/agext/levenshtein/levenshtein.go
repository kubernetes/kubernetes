// Copyright 2016 ALRUX Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Package levenshtein implements distance and similarity metrics for strings, based on the Levenshtein measure.

The Levenshtein `Distance` between two strings is the minimum total cost of edits that would convert the first string into the second. The allowed edit operations are insertions, deletions, and substitutions, all at character (one UTF-8 code point) level. Each operation has a default cost of 1, but each can be assigned its own cost equal to or greater than 0.

A `Distance` of 0 means the two strings are identical, and the higher the value the more different the strings. Since in practice we are interested in finding if the two strings are "close enough", it often does not make sense to continue the calculation once the result is mathematically guaranteed to exceed a desired threshold. Providing this value to the `Distance` function allows it to take a shortcut and return a lower bound instead of an exact cost when the threshold is exceeded.

The `Similarity` function calculates the distance, then converts it into a normalized metric within the range 0..1, with 1 meaning the strings are identical, and 0 that they have nothing in common. A minimum similarity threshold can be provided to speed up the calculation of the metric for strings that are far too dissimilar for the purpose at hand. All values under this threshold are rounded down to 0.

The `Match` function provides a similarity metric, with the same range and meaning as `Similarity`, but with a bonus for string pairs that share a common prefix and have a similarity above a "bonus threshold". It uses the same method as proposed by Winkler for the Jaro distance, and the reasoning behind it is that these string pairs are very likely spelling variations or errors, and they are more closely linked than the edit distance alone would suggest.

The underlying `Calculate` function is also exported, to allow the building of other derivative metrics, if needed.
*/
package levenshtein

// Calculate determines the Levenshtein distance between two strings, using
// the given costs for each edit operation. It returns the distance along with
// the lengths of the longest common prefix and suffix.
//
// If maxCost is non-zero, the calculation stops as soon as the distance is determined
// to be greater than maxCost. Therefore, any return value higher than maxCost is a
// lower bound for the actual distance.
func Calculate(str1, str2 []rune, maxCost, insCost, subCost, delCost int) (dist, prefixLen, suffixLen int) {
	l1, l2 := len(str1), len(str2)
	// trim common prefix, if any, as it doesn't affect the distance
	for ; prefixLen < l1 && prefixLen < l2; prefixLen++ {
		if str1[prefixLen] != str2[prefixLen] {
			break
		}
	}
	str1, str2 = str1[prefixLen:], str2[prefixLen:]
	l1 -= prefixLen
	l2 -= prefixLen
	// trim common suffix, if any, as it doesn't affect the distance
	for 0 < l1 && 0 < l2 {
		if str1[l1-1] != str2[l2-1] {
			str1, str2 = str1[:l1], str2[:l2]
			break
		}
		l1--
		l2--
		suffixLen++
	}
	// if the first string is empty, the distance is the length of the second string times the cost of insertion
	if l1 == 0 {
		dist = l2 * insCost
		return
	}
	// if the second string is empty, the distance is the length of the first string times the cost of deletion
	if l2 == 0 {
		dist = l1 * delCost
		return
	}

	// variables used in inner "for" loops
	var y, dy, c, l int

	// if maxCost is greater than or equal to the maximum possible distance, it's equivalent to 'unlimited'
	if maxCost > 0 {
		if subCost < delCost+insCost {
			if maxCost >= l1*subCost+(l2-l1)*insCost {
				maxCost = 0
			}
		} else {
			if maxCost >= l1*delCost+l2*insCost {
				maxCost = 0
			}
		}
	}

	if maxCost > 0 {
		// prefer the longer string first, to minimize time;
		// a swap also transposes the meanings of insertion and deletion.
		if l1 < l2 {
			str1, str2, l1, l2, insCost, delCost = str2, str1, l2, l1, delCost, insCost
		}

		// the length differential times cost of deletion is a lower bound for the cost;
		// if it is higher than the maxCost, there is no point going into the main calculation.
		if dist = (l1 - l2) * delCost; dist > maxCost {
			return
		}

		d := make([]int, l1+1)

		// offset and length of d in the current row
		doff, dlen := 0, 1
		for y, dy = 1, delCost; y <= l1 && dy <= maxCost; dlen++ {
			d[y] = dy
			y++
			dy = y * delCost
		}
		// fmt.Printf("%q -> %q: init doff=%d dlen=%d d[%d:%d]=%v\n", str1, str2, doff, dlen, doff, doff+dlen, d[doff:doff+dlen])

		for x := 0; x < l2; x++ {
			dy, d[doff] = d[doff], d[doff]+insCost
			for d[doff] > maxCost && dlen > 0 {
				if str1[doff] != str2[x] {
					dy += subCost
				}
				doff++
				dlen--
				if c = d[doff] + insCost; c < dy {
					dy = c
				}
				dy, d[doff] = d[doff], dy
			}
			for y, l = doff, doff+dlen-1; y < l; dy, d[y] = d[y], dy {
				if str1[y] != str2[x] {
					dy += subCost
				}
				if c = d[y] + delCost; c < dy {
					dy = c
				}
				y++
				if c = d[y] + insCost; c < dy {
					dy = c
				}
			}
			if y < l1 {
				if str1[y] != str2[x] {
					dy += subCost
				}
				if c = d[y] + delCost; c < dy {
					dy = c
				}
				for ; dy <= maxCost && y < l1; dy, d[y] = dy+delCost, dy {
					y++
					dlen++
				}
			}
			// fmt.Printf("%q -> %q: x=%d doff=%d dlen=%d d[%d:%d]=%v\n", str1, str2, x, doff, dlen, doff, doff+dlen, d[doff:doff+dlen])
			if dlen == 0 {
				dist = maxCost + 1
				return
			}
		}
		if doff+dlen-1 < l1 {
			dist = maxCost + 1
			return
		}
		dist = d[l1]
	} else {
		// ToDo: This is O(l1*l2) time and O(min(l1,l2)) space; investigate if it is
		// worth to implement diagonal approach - O(l1*(1+dist)) time, up to O(l1*l2) space
		// http://www.csse.monash.edu.au/~lloyd/tildeStrings/Alignment/92.IPL.html

		// prefer the shorter string first, to minimize space; time is O(l1*l2) anyway;
		// a swap also transposes the meanings of insertion and deletion.
		if l1 > l2 {
			str1, str2, l1, l2, insCost, delCost = str2, str1, l2, l1, delCost, insCost
		}
		d := make([]int, l1+1)

		for y = 1; y <= l1; y++ {
			d[y] = y * delCost
		}
		for x := 0; x < l2; x++ {
			dy, d[0] = d[0], d[0]+insCost
			for y = 0; y < l1; dy, d[y] = d[y], dy {
				if str1[y] != str2[x] {
					dy += subCost
				}
				if c = d[y] + delCost; c < dy {
					dy = c
				}
				y++
				if c = d[y] + insCost; c < dy {
					dy = c
				}
			}
		}
		dist = d[l1]
	}

	return
}

// Distance returns the Levenshtein distance between str1 and str2, using the
// default or provided cost values. Pass nil for the third argument to use the
// default cost of 1 for all three operations, with no maximum.
func Distance(str1, str2 string, p *Params) int {
	if p == nil {
		p = defaultParams
	}
	dist, _, _ := Calculate([]rune(str1), []rune(str2), p.maxCost, p.insCost, p.subCost, p.delCost)
	return dist
}

// Similarity returns a score in the range of 0..1 for how similar the two strings are.
// A score of 1 means the strings are identical, and 0 means they have nothing in common.
//
// A nil third argument uses the default cost of 1 for all three operations.
//
// If a non-zero MinScore value is provided in the parameters, scores lower than it
// will be returned as 0.
func Similarity(str1, str2 string, p *Params) float64 {
	return Match(str1, str2, p.Clone().BonusThreshold(1.1)) // guaranteed no bonus
}

// Match returns a similarity score adjusted by the same method as proposed by Winkler for
// the Jaro distance - giving a bonus to string pairs that share a common prefix, only if their
// similarity score is already over a threshold.
//
// The score is in the range of 0..1, with 1 meaning the strings are identical,
// and 0 meaning they have nothing in common.
//
// A nil third argument uses the default cost of 1 for all three operations, maximum length of
// common prefix to consider for bonus of 4, scaling factor of 0.1, and bonus threshold of 0.7.
//
// If a non-zero MinScore value is provided in the parameters, scores lower than it
// will be returned as 0.
func Match(str1, str2 string, p *Params) float64 {
	s1, s2 := []rune(str1), []rune(str2)
	l1, l2 := len(s1), len(s2)
	// two empty strings are identical; shortcut also avoids divByZero issues later on.
	if l1 == 0 && l2 == 0 {
		return 1
	}

	if p == nil {
		p = defaultParams
	}

	// a min over 1 can never be satisfied, so the score is 0.
	if p.minScore > 1 {
		return 0
	}

	insCost, delCost, maxDist, max := p.insCost, p.delCost, 0, 0
	if l1 > l2 {
		l1, l2, insCost, delCost = l2, l1, delCost, insCost
	}

	if p.subCost < delCost+insCost {
		maxDist = l1*p.subCost + (l2-l1)*insCost
	} else {
		maxDist = l1*delCost + l2*insCost
	}

	// a zero min is always satisfied, so no need to set a max cost.
	if p.minScore > 0 {
		// if p.minScore is lower than p.bonusThreshold, we can use a simplified formula
		// for the max cost, because a sim score below min cannot receive a bonus.
		if p.minScore < p.bonusThreshold {
			// round down the max - a cost equal to a rounded up max would already be under min.
			max = int((1 - p.minScore) * float64(maxDist))
		} else {
			// p.minScore <= sim + p.bonusPrefix*p.bonusScale*(1-sim)
			// p.minScore <= (1-dist/maxDist) + p.bonusPrefix*p.bonusScale*(1-(1-dist/maxDist))
			// p.minScore <= 1 - dist/maxDist + p.bonusPrefix*p.bonusScale*dist/maxDist
			// 1 - p.minScore >= dist/maxDist - p.bonusPrefix*p.bonusScale*dist/maxDist
			// (1-p.minScore)*maxDist/(1-p.bonusPrefix*p.bonusScale) >= dist
			max = int((1 - p.minScore) * float64(maxDist) / (1 - float64(p.bonusPrefix)*p.bonusScale))
		}
	}

	dist, pl, _ := Calculate(s1, s2, max, p.insCost, p.subCost, p.delCost)
	if max > 0 && dist > max {
		return 0
	}
	sim := 1 - float64(dist)/float64(maxDist)

	if sim >= p.bonusThreshold && sim < 1 && p.bonusPrefix > 0 && p.bonusScale > 0 {
		if pl > p.bonusPrefix {
			pl = p.bonusPrefix
		}
		sim += float64(pl) * p.bonusScale * (1 - sim)
	}

	if sim < p.minScore {
		return 0
	}

	return sim
}
