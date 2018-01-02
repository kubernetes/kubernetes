package influxql

// linearFloat computes the the slope of the line between the points (previousTime, previousValue) and (nextTime, nextValue)
// and returns the value of the point on the line with time windowTime
// y = mx + b
func linearFloat(windowTime, previousTime, nextTime int64, previousValue, nextValue float64) float64 {
	m := (nextValue - previousValue) / float64(nextTime-previousTime) // the slope of the line
	x := float64(windowTime - previousTime)                           // how far into the interval we are
	b := previousValue
	return m*x + b
}

// linearInteger computes the the slope of the line between the points (previousTime, previousValue) and (nextTime, nextValue)
// and returns the value of the point on the line with time windowTime
// y = mx + b
func linearInteger(windowTime, previousTime, nextTime int64, previousValue, nextValue int64) int64 {
	m := float64(nextValue-previousValue) / float64(nextTime-previousTime) // the slope of the line
	x := float64(windowTime - previousTime)                                // how far into the interval we are
	b := float64(previousValue)
	return int64(m*x + b)
}
