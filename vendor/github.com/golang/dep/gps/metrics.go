// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"bytes"
	"fmt"
	"log"
	"sort"
	"text/tabwriter"
	"time"
)

type metrics struct {
	stack []string
	times map[string]time.Duration
	last  time.Time
}

func newMetrics() *metrics {
	return &metrics{
		stack: []string{"other"},
		times: map[string]time.Duration{
			"other": 0,
		},
		last: time.Now(),
	}
}

func (m *metrics) push(name string) {
	cn := m.stack[len(m.stack)-1]
	m.times[cn] = m.times[cn] + time.Since(m.last)

	m.stack = append(m.stack, name)
	m.last = time.Now()
}

func (m *metrics) pop() {
	on := m.stack[len(m.stack)-1]
	m.times[on] = m.times[on] + time.Since(m.last)

	m.stack = m.stack[:len(m.stack)-1]
	m.last = time.Now()
}

func (m *metrics) dump(l *log.Logger) {
	s := make(ndpairs, len(m.times))
	k := 0
	for n, d := range m.times {
		s[k] = ndpair{
			n: n,
			d: d,
		}
		k++
	}

	sort.Sort(sort.Reverse(s))

	var tot time.Duration
	var buf bytes.Buffer
	w := tabwriter.NewWriter(&buf, 0, 0, 1, ' ', tabwriter.AlignRight)
	for _, nd := range s {
		tot += nd.d
		fmt.Fprintf(w, "\t%s:\t%v\t\n", nd.n, nd.d)
	}
	fmt.Fprintf(w, "\n\tTOTAL:\t%v\t\n", tot)
	w.Flush()

	l.Println("\nSolver wall times by segment:")
	l.Println((&buf).String())
}

type ndpair struct {
	n string
	d time.Duration
}

type ndpairs []ndpair

func (s ndpairs) Less(i, j int) bool { return s[i].d < s[j].d }
func (s ndpairs) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s ndpairs) Len() int           { return len(s) }
