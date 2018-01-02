package influxql

import (
	"math"
	"time"

	"github.com/influxdata/influxdb/influxql/neldermead"
)

// FloatMeanReducer calculates the mean of the aggregated points.
type FloatMeanReducer struct {
	sum   float64
	count uint32
}

// NewFloatMeanReducer creates a new FloatMeanReducer.
func NewFloatMeanReducer() *FloatMeanReducer {
	return &FloatMeanReducer{}
}

// AggregateFloat aggregates a point into the reducer.
func (r *FloatMeanReducer) AggregateFloat(p *FloatPoint) {
	if p.Aggregated >= 2 {
		r.sum += p.Value * float64(p.Aggregated)
		r.count += p.Aggregated
	} else {
		r.sum += p.Value
		r.count++
	}
}

// Emit emits the mean of the aggregated points as a single point.
func (r *FloatMeanReducer) Emit() []FloatPoint {
	return []FloatPoint{{
		Time:       ZeroTime,
		Value:      r.sum / float64(r.count),
		Aggregated: r.count,
	}}
}

// IntegerMeanReducer calculates the mean of the aggregated points.
type IntegerMeanReducer struct {
	sum   int64
	count uint32
}

// NewIntegerMeanReducer creates a new IntegerMeanReducer.
func NewIntegerMeanReducer() *IntegerMeanReducer {
	return &IntegerMeanReducer{}
}

// AggregateInteger aggregates a point into the reducer.
func (r *IntegerMeanReducer) AggregateInteger(p *IntegerPoint) {
	if p.Aggregated >= 2 {
		r.sum += p.Value * int64(p.Aggregated)
		r.count += p.Aggregated
	} else {
		r.sum += p.Value
		r.count++
	}
}

// Emit emits the mean of the aggregated points as a single point.
func (r *IntegerMeanReducer) Emit() []FloatPoint {
	return []FloatPoint{{
		Time:       ZeroTime,
		Value:      float64(r.sum) / float64(r.count),
		Aggregated: r.count,
	}}
}

// FloatDerivativeReducer calculates the derivative of the aggregated points.
type FloatDerivativeReducer struct {
	interval      Interval
	prev          FloatPoint
	curr          FloatPoint
	isNonNegative bool
	ascending     bool
}

// NewFloatDerivativeReducer creates a new FloatDerivativeReducer.
func NewFloatDerivativeReducer(interval Interval, isNonNegative, ascending bool) *FloatDerivativeReducer {
	return &FloatDerivativeReducer{
		interval:      interval,
		isNonNegative: isNonNegative,
		ascending:     ascending,
		prev:          FloatPoint{Nil: true},
		curr:          FloatPoint{Nil: true},
	}
}

// AggregateFloat aggregates a point into the reducer and updates the current window.
func (r *FloatDerivativeReducer) AggregateFloat(p *FloatPoint) {
	// Skip past a point when it does not advance the stream. A joined series
	// may have multiple points at the same time so we will discard anything
	// except the first point we encounter.
	if !r.curr.Nil && r.curr.Time == p.Time {
		return
	}

	r.prev = r.curr
	r.curr = *p
}

// Emit emits the derivative of the reducer at the current point.
func (r *FloatDerivativeReducer) Emit() []FloatPoint {
	if !r.prev.Nil {
		// Calculate the derivative of successive points by dividing the
		// difference of each value by the elapsed time normalized to the interval.
		diff := r.curr.Value - r.prev.Value
		elapsed := r.curr.Time - r.prev.Time
		if !r.ascending {
			elapsed = -elapsed
		}
		value := diff / (float64(elapsed) / float64(r.interval.Duration))

		// Mark this point as read by changing the previous point to nil.
		r.prev.Nil = true

		// Drop negative values for non-negative derivatives.
		if r.isNonNegative && diff < 0 {
			return nil
		}
		return []FloatPoint{{Time: r.curr.Time, Value: value}}
	}
	return nil
}

// IntegerDerivativeReducer calculates the derivative of the aggregated points.
type IntegerDerivativeReducer struct {
	interval      Interval
	prev          IntegerPoint
	curr          IntegerPoint
	isNonNegative bool
	ascending     bool
}

// NewIntegerDerivativeReducer creates a new IntegerDerivativeReducer.
func NewIntegerDerivativeReducer(interval Interval, isNonNegative, ascending bool) *IntegerDerivativeReducer {
	return &IntegerDerivativeReducer{
		interval:      interval,
		isNonNegative: isNonNegative,
		ascending:     ascending,
		prev:          IntegerPoint{Nil: true},
		curr:          IntegerPoint{Nil: true},
	}
}

// AggregateInteger aggregates a point into the reducer and updates the current window.
func (r *IntegerDerivativeReducer) AggregateInteger(p *IntegerPoint) {
	// Skip past a point when it does not advance the stream. A joined series
	// may have multiple points at the same time so we will discard anything
	// except the first point we encounter.
	if !r.curr.Nil && r.curr.Time == p.Time {
		return
	}

	r.prev = r.curr
	r.curr = *p
}

// Emit emits the derivative of the reducer at the current point.
func (r *IntegerDerivativeReducer) Emit() []FloatPoint {
	if !r.prev.Nil {
		// Calculate the derivative of successive points by dividing the
		// difference of each value by the elapsed time normalized to the interval.
		diff := float64(r.curr.Value - r.prev.Value)
		elapsed := r.curr.Time - r.prev.Time
		if !r.ascending {
			elapsed = -elapsed
		}
		value := diff / (float64(elapsed) / float64(r.interval.Duration))

		// Mark this point as read by changing the previous point to nil.
		r.prev.Nil = true

		// Drop negative values for non-negative derivatives.
		if r.isNonNegative && diff < 0 {
			return nil
		}
		return []FloatPoint{{Time: r.curr.Time, Value: value}}
	}
	return nil
}

// FloatDifferenceReducer calculates the derivative of the aggregated points.
type FloatDifferenceReducer struct {
	prev FloatPoint
	curr FloatPoint
}

// NewFloatDifferenceReducer creates a new FloatDifferenceReducer.
func NewFloatDifferenceReducer() *FloatDifferenceReducer {
	return &FloatDifferenceReducer{
		prev: FloatPoint{Nil: true},
		curr: FloatPoint{Nil: true},
	}
}

// AggregateFloat aggregates a point into the reducer and updates the current window.
func (r *FloatDifferenceReducer) AggregateFloat(p *FloatPoint) {
	// Skip past a point when it does not advance the stream. A joined series
	// may have multiple points at the same time so we will discard anything
	// except the first point we encounter.
	if !r.curr.Nil && r.curr.Time == p.Time {
		return
	}

	r.prev = r.curr
	r.curr = *p
}

// Emit emits the difference of the reducer at the current point.
func (r *FloatDifferenceReducer) Emit() []FloatPoint {
	if !r.prev.Nil {
		// Calculate the difference of successive points.
		value := r.curr.Value - r.prev.Value

		// Mark this point as read by changing the previous point to nil.
		r.prev.Nil = true
		return []FloatPoint{{Time: r.curr.Time, Value: value}}
	}
	return nil
}

// IntegerDifferenceReducer calculates the derivative of the aggregated points.
type IntegerDifferenceReducer struct {
	prev IntegerPoint
	curr IntegerPoint
}

// NewIntegerDifferenceReducer creates a new IntegerDifferenceReducer.
func NewIntegerDifferenceReducer() *IntegerDifferenceReducer {
	return &IntegerDifferenceReducer{
		prev: IntegerPoint{Nil: true},
		curr: IntegerPoint{Nil: true},
	}
}

// AggregateInteger aggregates a point into the reducer and updates the current window.
func (r *IntegerDifferenceReducer) AggregateInteger(p *IntegerPoint) {
	// Skip past a point when it does not advance the stream. A joined series
	// may have multiple points at the same time so we will discard anything
	// except the first point we encounter.
	if !r.curr.Nil && r.curr.Time == p.Time {
		return
	}

	r.prev = r.curr
	r.curr = *p
}

// Emit emits the difference of the reducer at the current point.
func (r *IntegerDifferenceReducer) Emit() []IntegerPoint {
	if !r.prev.Nil {
		// Calculate the difference of successive points.
		value := r.curr.Value - r.prev.Value

		// Mark this point as read by changing the previous point to nil.
		r.prev.Nil = true
		return []IntegerPoint{{Time: r.curr.Time, Value: value}}
	}
	return nil
}

// FloatMovingAverageReducer calculates the moving average of the aggregated points.
type FloatMovingAverageReducer struct {
	pos  int
	sum  float64
	time int64
	buf  []float64
}

// NewFloatMovingAverageReducer creates a new FloatMovingAverageReducer.
func NewFloatMovingAverageReducer(n int) *FloatMovingAverageReducer {
	return &FloatMovingAverageReducer{
		buf: make([]float64, 0, n),
	}
}

// AggregateFloat aggregates a point into the reducer and updates the current window.
func (r *FloatMovingAverageReducer) AggregateFloat(p *FloatPoint) {
	if len(r.buf) != cap(r.buf) {
		r.buf = append(r.buf, p.Value)
	} else {
		r.sum -= r.buf[r.pos]
		r.buf[r.pos] = p.Value
	}
	r.sum += p.Value
	r.time = p.Time
	r.pos++
	if r.pos >= cap(r.buf) {
		r.pos = 0
	}
}

// Emit emits the moving average of the current window. Emit should be called
// after every call to AggregateFloat and it will produce one point if there
// is enough data to fill a window, otherwise it will produce zero points.
func (r *FloatMovingAverageReducer) Emit() []FloatPoint {
	if len(r.buf) != cap(r.buf) {
		return []FloatPoint{}
	}
	return []FloatPoint{
		{
			Value:      r.sum / float64(len(r.buf)),
			Time:       r.time,
			Aggregated: uint32(len(r.buf)),
		},
	}
}

// IntegerMovingAverageReducer calculates the moving average of the aggregated points.
type IntegerMovingAverageReducer struct {
	pos  int
	sum  int64
	time int64
	buf  []int64
}

// NewIntegerMovingAverageReducer creates a new IntegerMovingAverageReducer.
func NewIntegerMovingAverageReducer(n int) *IntegerMovingAverageReducer {
	return &IntegerMovingAverageReducer{
		buf: make([]int64, 0, n),
	}
}

// AggregateInteger aggregates a point into the reducer and updates the current window.
func (r *IntegerMovingAverageReducer) AggregateInteger(p *IntegerPoint) {
	if len(r.buf) != cap(r.buf) {
		r.buf = append(r.buf, p.Value)
	} else {
		r.sum -= r.buf[r.pos]
		r.buf[r.pos] = p.Value
	}
	r.sum += p.Value
	r.time = p.Time
	r.pos++
	if r.pos >= cap(r.buf) {
		r.pos = 0
	}
}

// Emit emits the moving average of the current window. Emit should be called
// after every call to AggregateInteger and it will produce one point if there
// is enough data to fill a window, otherwise it will produce zero points.
func (r *IntegerMovingAverageReducer) Emit() []FloatPoint {
	if len(r.buf) != cap(r.buf) {
		return []FloatPoint{}
	}
	return []FloatPoint{
		{
			Value:      float64(r.sum) / float64(len(r.buf)),
			Time:       r.time,
			Aggregated: uint32(len(r.buf)),
		},
	}
}

// FloatCumulativeSumReducer cumulates the values from each point.
type FloatCumulativeSumReducer struct {
	curr FloatPoint
}

// NewFloatCumulativeSumReducer creates a new FloatCumulativeSumReducer.
func NewFloatCumulativeSumReducer() *FloatCumulativeSumReducer {
	return &FloatCumulativeSumReducer{
		curr: FloatPoint{Nil: true},
	}
}

func (r *FloatCumulativeSumReducer) AggregateFloat(p *FloatPoint) {
	r.curr.Value += p.Value
	r.curr.Time = p.Time
	r.curr.Nil = false
}

func (r *FloatCumulativeSumReducer) Emit() []FloatPoint {
	var pts []FloatPoint
	if !r.curr.Nil {
		pts = []FloatPoint{r.curr}
	}
	return pts
}

// IntegerCumulativeSumReducer cumulates the values from each point.
type IntegerCumulativeSumReducer struct {
	curr IntegerPoint
}

// NewIntegerCumulativeSumReducer creates a new IntegerCumulativeSumReducer.
func NewIntegerCumulativeSumReducer() *IntegerCumulativeSumReducer {
	return &IntegerCumulativeSumReducer{
		curr: IntegerPoint{Nil: true},
	}
}

func (r *IntegerCumulativeSumReducer) AggregateInteger(p *IntegerPoint) {
	r.curr.Value += p.Value
	r.curr.Time = p.Time
	r.curr.Nil = false
}

func (r *IntegerCumulativeSumReducer) Emit() []IntegerPoint {
	var pts []IntegerPoint
	if !r.curr.Nil {
		pts = []IntegerPoint{r.curr}
	}
	return pts
}

// FloatHoltWintersReducer forecasts a series into the future.
// This is done using the Holt-Winters damped method.
//    1. Using the series the initial values are calculated using a SSE.
//    2. The series is forecasted into the future using the iterative relations.
type FloatHoltWintersReducer struct {
	// Season period
	m        int
	seasonal bool

	// Horizon
	h int

	// Interval between points
	interval int64
	// interval / 2 -- used to perform rounding
	halfInterval int64

	// Whether to include all data or only future values
	includeFitData bool

	// NelderMead optimizer
	optim *neldermead.Optimizer
	// Small difference bound for the optimizer
	epsilon float64

	y      []float64
	points []FloatPoint
}

const (
	// Arbitrary weight for initializing some intial guesses.
	// This should be in the  range [0,1]
	hwWeight = 0.5
	// Epsilon value for the minimization process
	hwDefaultEpsilon = 1.0e-4
	// Define a grid of initial guesses for the parameters: alpha, beta, gamma, and phi.
	// Keep in mind that this grid is N^4 so we should keep N small
	// The starting lower guess
	hwGuessLower = 0.3
	//  The upper bound on the grid
	hwGuessUpper = 1.0
	// The step between guesses
	hwGuessStep = 0.4
)

// NewFloatHoltWintersReducer creates a new FloatHoltWintersReducer.
func NewFloatHoltWintersReducer(h, m int, includeFitData bool, interval time.Duration) *FloatHoltWintersReducer {
	seasonal := true
	if m < 2 {
		seasonal = false
	}
	return &FloatHoltWintersReducer{
		h:              h,
		m:              m,
		seasonal:       seasonal,
		includeFitData: includeFitData,
		interval:       int64(interval),
		halfInterval:   int64(interval) / 2,
		optim:          neldermead.New(),
		epsilon:        hwDefaultEpsilon,
	}
}

func (r *FloatHoltWintersReducer) aggregate(time int64, value float64) {
	r.points = append(r.points, FloatPoint{
		Time:  time,
		Value: value,
	})
}

// AggregateFloat aggregates a point into the reducer and updates the current window.
func (r *FloatHoltWintersReducer) AggregateFloat(p *FloatPoint) {
	r.aggregate(p.Time, p.Value)
}

// AggregateInteger aggregates a point into the reducer and updates the current window.
func (r *FloatHoltWintersReducer) AggregateInteger(p *IntegerPoint) {
	r.aggregate(p.Time, float64(p.Value))
}

func (r *FloatHoltWintersReducer) roundTime(t int64) int64 {
	// Overflow safe round function
	remainder := t % r.interval
	if remainder > r.halfInterval {
		// Round up
		return (t/r.interval + 1) * r.interval
	}
	// Round down
	return (t / r.interval) * r.interval
}

// Emit returns the points generated by the HoltWinters algorithm.
func (r *FloatHoltWintersReducer) Emit() []FloatPoint {
	if l := len(r.points); l < 2 || r.seasonal && l < r.m || r.h <= 0 {
		return nil
	}
	// First fill in r.y with values and NaNs for missing values
	start, stop := r.roundTime(r.points[0].Time), r.roundTime(r.points[len(r.points)-1].Time)
	count := (stop - start) / r.interval
	if count <= 0 {
		return nil
	}
	r.y = make([]float64, 1, count)
	r.y[0] = r.points[0].Value
	t := r.roundTime(r.points[0].Time)
	for _, p := range r.points[1:] {
		rounded := r.roundTime(p.Time)
		if rounded <= t {
			// Drop values that occur for the same time bucket
			continue
		}
		t += r.interval
		// Add any missing values before the next point
		for rounded != t {
			// Add in a NaN so we can skip it later.
			r.y = append(r.y, math.NaN())
			t += r.interval
		}
		r.y = append(r.y, p.Value)
	}

	// Seasonality
	m := r.m

	// Starting guesses
	// NOTE: Since these values are guesses
	// in the cases where we were missing data,
	// we can just skip the value and call it good.

	l0 := 0.0
	if r.seasonal {
		for i := 0; i < m; i++ {
			if !math.IsNaN(r.y[i]) {
				l0 += (1 / float64(m)) * r.y[i]
			}
		}
	} else {
		l0 += hwWeight * r.y[0]
	}

	b0 := 0.0
	if r.seasonal {
		for i := 0; i < m && m+i < len(r.y); i++ {
			if !math.IsNaN(r.y[i]) && !math.IsNaN(r.y[m+i]) {
				b0 += 1 / float64(m*m) * (r.y[m+i] - r.y[i])
			}
		}
	} else {
		if !math.IsNaN(r.y[1]) {
			b0 = hwWeight * (r.y[1] - r.y[0])
		}
	}

	var s []float64
	if r.seasonal {
		s = make([]float64, m)
		for i := 0; i < m; i++ {
			if !math.IsNaN(r.y[i]) {
				s[i] = r.y[i] / l0
			} else {
				s[i] = 0
			}
		}
	}

	parameters := make([]float64, 6+len(s))
	parameters[4] = l0
	parameters[5] = b0
	o := len(parameters) - len(s)
	for i := range s {
		parameters[i+o] = s[i]
	}

	// Determine best fit for the various parameters
	minSSE := math.Inf(1)
	var bestParams []float64
	for alpha := hwGuessLower; alpha < hwGuessUpper; alpha += hwGuessStep {
		for beta := hwGuessLower; beta < hwGuessUpper; beta += hwGuessStep {
			for gamma := hwGuessLower; gamma < hwGuessUpper; gamma += hwGuessStep {
				for phi := hwGuessLower; phi < hwGuessUpper; phi += hwGuessStep {
					parameters[0] = alpha
					parameters[1] = beta
					parameters[2] = gamma
					parameters[3] = phi
					sse, params := r.optim.Optimize(r.sse, parameters, r.epsilon, 1)
					if sse < minSSE || bestParams == nil {
						minSSE = sse
						bestParams = params
					}
				}
			}
		}
	}

	// Forecast
	forecasted := r.forecast(r.h, bestParams)
	var points []FloatPoint
	if r.includeFitData {
		start := r.points[0].Time
		points = make([]FloatPoint, 0, len(forecasted))
		for i, v := range forecasted {
			if !math.IsNaN(v) {
				t := start + r.interval*(int64(i))
				points = append(points, FloatPoint{
					Value: v,
					Time:  t,
				})
			}
		}
	} else {
		stop := r.points[len(r.points)-1].Time
		points = make([]FloatPoint, 0, r.h)
		for i, v := range forecasted[len(r.y):] {
			if !math.IsNaN(v) {
				t := stop + r.interval*(int64(i)+1)
				points = append(points, FloatPoint{
					Value: v,
					Time:  t,
				})
			}
		}
	}
	// Clear data set
	r.y = r.y[0:0]
	return points
}

// Using the recursive relations compute the next values
func (r *FloatHoltWintersReducer) next(alpha, beta, gamma, phi, phiH, yT, lTp, bTp, sTm, sTmh float64) (yTh, lT, bT, sT float64) {
	lT = alpha*(yT/sTm) + (1-alpha)*(lTp+phi*bTp)
	bT = beta*(lT-lTp) + (1-beta)*phi*bTp
	sT = gamma*(yT/(lTp+phi*bTp)) + (1-gamma)*sTm
	yTh = (lT + phiH*bT) * sTmh
	return
}

// Forecast the data h points into the future.
func (r *FloatHoltWintersReducer) forecast(h int, params []float64) []float64 {
	// Constrain parameters
	r.constrain(params)

	yT := r.y[0]

	phi := params[3]
	phiH := phi

	lT := params[4]
	bT := params[5]
	sT := 0.0

	// seasonals is a ring buffer of past sT values
	var seasonals []float64
	var m, so int
	if r.seasonal {
		seasonals = params[6:]
		m = len(params[6:])
		if m == 1 {
			seasonals[0] = 1
		}
		// Season index offset
		so = m - 1
	}

	forecasted := make([]float64, len(r.y)+h)
	forecasted[0] = yT
	l := len(r.y)
	var hm int
	stm, stmh := 1.0, 1.0
	for t := 1; t < l+h; t++ {
		if r.seasonal {
			hm = t % m
			stm = seasonals[(t-m+so)%m]
			stmh = seasonals[(t-m+hm+so)%m]
		}
		yT, lT, bT, sT = r.next(
			params[0], // alpha
			params[1], // beta
			params[2], // gamma
			phi,
			phiH,
			yT,
			lT,
			bT,
			stm,
			stmh,
		)
		phiH += math.Pow(phi, float64(t))

		if r.seasonal {
			seasonals[(t+so)%m] = sT
			so++
		}

		forecasted[t] = yT
	}
	return forecasted
}

// Compute sum squared error for the given parameters.
func (r *FloatHoltWintersReducer) sse(params []float64) float64 {
	sse := 0.0
	forecasted := r.forecast(0, params)
	for i := range forecasted {
		// Skip missing values since we cannot use them to compute an error.
		if !math.IsNaN(r.y[i]) {
			// Compute error
			if math.IsNaN(forecasted[i]) {
				// Penalize forecasted NaNs
				return math.Inf(1)
			}
			diff := forecasted[i] - r.y[i]
			sse += diff * diff
		}
	}
	return sse
}

// Constrain alpha, beta, gamma, phi in the range [0, 1]
func (r *FloatHoltWintersReducer) constrain(x []float64) {
	// alpha
	if x[0] > 1 {
		x[0] = 1
	}
	if x[0] < 0 {
		x[0] = 0
	}
	// beta
	if x[1] > 1 {
		x[1] = 1
	}
	if x[1] < 0 {
		x[1] = 0
	}
	// gamma
	if x[2] > 1 {
		x[2] = 1
	}
	if x[2] < 0 {
		x[2] = 0
	}
	// phi
	if x[3] > 1 {
		x[3] = 1
	}
	if x[3] < 0 {
		x[3] = 0
	}
}
