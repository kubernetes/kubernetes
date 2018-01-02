// Copyright 2010 Petar Maymounkov. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package llrb

import "math"

// avgVar maintains the average and variance of a stream of numbers
// in a space-efficient manner.
type avgVar struct {
	count      int64
	sum, sumsq float64
}

func (av *avgVar) Init() {
	av.count = 0
	av.sum = 0.0
	av.sumsq = 0.0
}

func (av *avgVar) Add(sample float64) {
	av.count++
	av.sum += sample
	av.sumsq += sample * sample
}

func (av *avgVar) GetCount() int64 { return av.count }

func (av *avgVar) GetAvg() float64 { return av.sum / float64(av.count) }

func (av *avgVar) GetTotal() float64 { return av.sum }

func (av *avgVar) GetVar() float64 {
	a := av.GetAvg()
	return av.sumsq/float64(av.count) - a*a
}

func (av *avgVar) GetStdDev() float64 { return math.Sqrt(av.GetVar()) }
