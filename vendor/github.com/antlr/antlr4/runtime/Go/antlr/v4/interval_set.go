// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"strconv"
	"strings"
)

type Interval struct {
	Start int
	Stop  int
}

/* stop is not included! */
func NewInterval(start, stop int) *Interval {
	i := new(Interval)

	i.Start = start
	i.Stop = stop
	return i
}

func (i *Interval) Contains(item int) bool {
	return item >= i.Start && item < i.Stop
}

func (i *Interval) String() string {
	if i.Start == i.Stop-1 {
		return strconv.Itoa(i.Start)
	}

	return strconv.Itoa(i.Start) + ".." + strconv.Itoa(i.Stop-1)
}

func (i *Interval) length() int {
	return i.Stop - i.Start
}

type IntervalSet struct {
	intervals []*Interval
	readOnly  bool
}

func NewIntervalSet() *IntervalSet {

	i := new(IntervalSet)

	i.intervals = nil
	i.readOnly = false

	return i
}

func (i *IntervalSet) first() int {
	if len(i.intervals) == 0 {
		return TokenInvalidType
	}

	return i.intervals[0].Start
}

func (i *IntervalSet) addOne(v int) {
	i.addInterval(NewInterval(v, v+1))
}

func (i *IntervalSet) addRange(l, h int) {
	i.addInterval(NewInterval(l, h+1))
}

func (i *IntervalSet) addInterval(v *Interval) {
	if i.intervals == nil {
		i.intervals = make([]*Interval, 0)
		i.intervals = append(i.intervals, v)
	} else {
		// find insert pos
		for k, interval := range i.intervals {
			// distinct range -> insert
			if v.Stop < interval.Start {
				i.intervals = append(i.intervals[0:k], append([]*Interval{v}, i.intervals[k:]...)...)
				return
			} else if v.Stop == interval.Start {
				i.intervals[k].Start = v.Start
				return
			} else if v.Start <= interval.Stop {
				i.intervals[k] = NewInterval(intMin(interval.Start, v.Start), intMax(interval.Stop, v.Stop))

				// if not applying to end, merge potential overlaps
				if k < len(i.intervals)-1 {
					l := i.intervals[k]
					r := i.intervals[k+1]
					// if r contained in l
					if l.Stop >= r.Stop {
						i.intervals = append(i.intervals[0:k+1], i.intervals[k+2:]...)
					} else if l.Stop >= r.Start { // partial overlap
						i.intervals[k] = NewInterval(l.Start, r.Stop)
						i.intervals = append(i.intervals[0:k+1], i.intervals[k+2:]...)
					}
				}
				return
			}
		}
		// greater than any exiting
		i.intervals = append(i.intervals, v)
	}
}

func (i *IntervalSet) addSet(other *IntervalSet) *IntervalSet {
	if other.intervals != nil {
		for k := 0; k < len(other.intervals); k++ {
			i2 := other.intervals[k]
			i.addInterval(NewInterval(i2.Start, i2.Stop))
		}
	}
	return i
}

func (i *IntervalSet) complement(start int, stop int) *IntervalSet {
	result := NewIntervalSet()
	result.addInterval(NewInterval(start, stop+1))
	for j := 0; j < len(i.intervals); j++ {
		result.removeRange(i.intervals[j])
	}
	return result
}

func (i *IntervalSet) contains(item int) bool {
	if i.intervals == nil {
		return false
	}
	for k := 0; k < len(i.intervals); k++ {
		if i.intervals[k].Contains(item) {
			return true
		}
	}
	return false
}

func (i *IntervalSet) length() int {
	len := 0

	for _, v := range i.intervals {
		len += v.length()
	}

	return len
}

func (i *IntervalSet) removeRange(v *Interval) {
	if v.Start == v.Stop-1 {
		i.removeOne(v.Start)
	} else if i.intervals != nil {
		k := 0
		for n := 0; n < len(i.intervals); n++ {
			ni := i.intervals[k]
			// intervals are ordered
			if v.Stop <= ni.Start {
				return
			} else if v.Start > ni.Start && v.Stop < ni.Stop {
				i.intervals[k] = NewInterval(ni.Start, v.Start)
				x := NewInterval(v.Stop, ni.Stop)
				// i.intervals.splice(k, 0, x)
				i.intervals = append(i.intervals[0:k], append([]*Interval{x}, i.intervals[k:]...)...)
				return
			} else if v.Start <= ni.Start && v.Stop >= ni.Stop {
				//                i.intervals.splice(k, 1)
				i.intervals = append(i.intervals[0:k], i.intervals[k+1:]...)
				k = k - 1 // need another pass
			} else if v.Start < ni.Stop {
				i.intervals[k] = NewInterval(ni.Start, v.Start)
			} else if v.Stop < ni.Stop {
				i.intervals[k] = NewInterval(v.Stop, ni.Stop)
			}
			k++
		}
	}
}

func (i *IntervalSet) removeOne(v int) {
	if i.intervals != nil {
		for k := 0; k < len(i.intervals); k++ {
			ki := i.intervals[k]
			// intervals i ordered
			if v < ki.Start {
				return
			} else if v == ki.Start && v == ki.Stop-1 {
				//				i.intervals.splice(k, 1)
				i.intervals = append(i.intervals[0:k], i.intervals[k+1:]...)
				return
			} else if v == ki.Start {
				i.intervals[k] = NewInterval(ki.Start+1, ki.Stop)
				return
			} else if v == ki.Stop-1 {
				i.intervals[k] = NewInterval(ki.Start, ki.Stop-1)
				return
			} else if v < ki.Stop-1 {
				x := NewInterval(ki.Start, v)
				ki.Start = v + 1
				//				i.intervals.splice(k, 0, x)
				i.intervals = append(i.intervals[0:k], append([]*Interval{x}, i.intervals[k:]...)...)
				return
			}
		}
	}
}

func (i *IntervalSet) String() string {
	return i.StringVerbose(nil, nil, false)
}

func (i *IntervalSet) StringVerbose(literalNames []string, symbolicNames []string, elemsAreChar bool) string {

	if i.intervals == nil {
		return "{}"
	} else if literalNames != nil || symbolicNames != nil {
		return i.toTokenString(literalNames, symbolicNames)
	} else if elemsAreChar {
		return i.toCharString()
	}

	return i.toIndexString()
}

func (i *IntervalSet) GetIntervals() []*Interval {
	return i.intervals
}

func (i *IntervalSet) toCharString() string {
	names := make([]string, len(i.intervals))

	var sb strings.Builder

	for j := 0; j < len(i.intervals); j++ {
		v := i.intervals[j]
		if v.Stop == v.Start+1 {
			if v.Start == TokenEOF {
				names = append(names, "<EOF>")
			} else {
				sb.WriteByte('\'')
				sb.WriteRune(rune(v.Start))
				sb.WriteByte('\'')
				names = append(names, sb.String())
				sb.Reset()
			}
		} else {
			sb.WriteByte('\'')
			sb.WriteRune(rune(v.Start))
			sb.WriteString("'..'")
			sb.WriteRune(rune(v.Stop - 1))
			sb.WriteByte('\'')
			names = append(names, sb.String())
			sb.Reset()
		}
	}
	if len(names) > 1 {
		return "{" + strings.Join(names, ", ") + "}"
	}

	return names[0]
}

func (i *IntervalSet) toIndexString() string {

	names := make([]string, 0)
	for j := 0; j < len(i.intervals); j++ {
		v := i.intervals[j]
		if v.Stop == v.Start+1 {
			if v.Start == TokenEOF {
				names = append(names, "<EOF>")
			} else {
				names = append(names, strconv.Itoa(v.Start))
			}
		} else {
			names = append(names, strconv.Itoa(v.Start)+".."+strconv.Itoa(v.Stop-1))
		}
	}
	if len(names) > 1 {
		return "{" + strings.Join(names, ", ") + "}"
	}

	return names[0]
}

func (i *IntervalSet) toTokenString(literalNames []string, symbolicNames []string) string {
	names := make([]string, 0)
	for _, v := range i.intervals {
		for j := v.Start; j < v.Stop; j++ {
			names = append(names, i.elementName(literalNames, symbolicNames, j))
		}
	}
	if len(names) > 1 {
		return "{" + strings.Join(names, ", ") + "}"
	}

	return names[0]
}

func (i *IntervalSet) elementName(literalNames []string, symbolicNames []string, a int) string {
	if a == TokenEOF {
		return "<EOF>"
	} else if a == TokenEpsilon {
		return "<EPSILON>"
	} else {
		if a < len(literalNames) && literalNames[a] != "" {
			return literalNames[a]
		}

		return symbolicNames[a]
	}
}
