package coordinate

import (
	"math"
	"math/rand"
	"time"
)

// Coordinate is a specialized structure for holding network coordinates for the
// Vivaldi-based coordinate mapping algorithm. All of the fields should be public
// to enable this to be serialized. All values in here are in units of seconds.
type Coordinate struct {
	// Vec is the Euclidean portion of the coordinate. This is used along
	// with the other fields to provide an overall distance estimate. The
	// units here are seconds.
	Vec []float64

	// Err reflects the confidence in the given coordinate and is updated
	// dynamically by the Vivaldi Client. This is dimensionless.
	Error float64

	// Adjustment is a distance offset computed based on a calculation over
	// observations from all other nodes over a fixed window and is updated
	// dynamically by the Vivaldi Client. The units here are seconds.
	Adjustment float64

	// Height is a distance offset that accounts for non-Euclidean effects
	// which model the access links from nodes to the core Internet. The access
	// links are usually set by bandwidth and congestion, and the core links
	// usually follow distance based on geography.
	Height float64
}

const (
	// secondsToNanoseconds is used to convert float seconds to nanoseconds.
	secondsToNanoseconds = 1.0e9

	// zeroThreshold is used to decide if two coordinates are on top of each
	// other.
	zeroThreshold = 1.0e-6
)

// ErrDimensionalityConflict will be panic-d if you try to perform operations
// with incompatible dimensions.
type DimensionalityConflictError struct{}

// Adds the error interface.
func (e DimensionalityConflictError) Error() string {
	return "coordinate dimensionality does not match"
}

// NewCoordinate creates a new coordinate at the origin, using the given config
// to supply key initial values.
func NewCoordinate(config *Config) *Coordinate {
	return &Coordinate{
		Vec:        make([]float64, config.Dimensionality),
		Error:      config.VivaldiErrorMax,
		Adjustment: 0.0,
		Height:     config.HeightMin,
	}
}

// Clone creates an independent copy of this coordinate.
func (c *Coordinate) Clone() *Coordinate {
	vec := make([]float64, len(c.Vec))
	copy(vec, c.Vec)
	return &Coordinate{
		Vec:        vec,
		Error:      c.Error,
		Adjustment: c.Adjustment,
		Height:     c.Height,
	}
}

// IsCompatibleWith checks to see if the two coordinates are compatible
// dimensionally. If this returns true then you are guaranteed to not get
// any runtime errors operating on them.
func (c *Coordinate) IsCompatibleWith(other *Coordinate) bool {
	return len(c.Vec) == len(other.Vec)
}

// ApplyForce returns the result of applying the force from the direction of the
// other coordinate.
func (c *Coordinate) ApplyForce(config *Config, force float64, other *Coordinate) *Coordinate {
	if !c.IsCompatibleWith(other) {
		panic(DimensionalityConflictError{})
	}

	ret := c.Clone()
	unit, mag := unitVectorAt(c.Vec, other.Vec)
	ret.Vec = add(ret.Vec, mul(unit, force))
	if mag > zeroThreshold {
		ret.Height = (ret.Height+other.Height)*force/mag + ret.Height
		ret.Height = math.Max(ret.Height, config.HeightMin)
	}
	return ret
}

// DistanceTo returns the distance between this coordinate and the other
// coordinate, including adjustments.
func (c *Coordinate) DistanceTo(other *Coordinate) time.Duration {
	if !c.IsCompatibleWith(other) {
		panic(DimensionalityConflictError{})
	}

	dist := c.rawDistanceTo(other)
	adjustedDist := dist + c.Adjustment + other.Adjustment
	if adjustedDist > 0.0 {
		dist = adjustedDist
	}
	return time.Duration(dist * secondsToNanoseconds)
}

// rawDistanceTo returns the Vivaldi distance between this coordinate and the
// other coordinate in seconds, not including adjustments. This assumes the
// dimensions have already been checked to be compatible.
func (c *Coordinate) rawDistanceTo(other *Coordinate) float64 {
	return magnitude(diff(c.Vec, other.Vec)) + c.Height + other.Height
}

// add returns the sum of vec1 and vec2. This assumes the dimensions have
// already been checked to be compatible.
func add(vec1 []float64, vec2 []float64) []float64 {
	ret := make([]float64, len(vec1))
	for i, _ := range ret {
		ret[i] = vec1[i] + vec2[i]
	}
	return ret
}

// diff returns the difference between the vec1 and vec2. This assumes the
// dimensions have already been checked to be compatible.
func diff(vec1 []float64, vec2 []float64) []float64 {
	ret := make([]float64, len(vec1))
	for i, _ := range ret {
		ret[i] = vec1[i] - vec2[i]
	}
	return ret
}

// mul returns vec multiplied by a scalar factor.
func mul(vec []float64, factor float64) []float64 {
	ret := make([]float64, len(vec))
	for i, _ := range vec {
		ret[i] = vec[i] * factor
	}
	return ret
}

// magnitude computes the magnitude of the vec.
func magnitude(vec []float64) float64 {
	sum := 0.0
	for i, _ := range vec {
		sum += vec[i] * vec[i]
	}
	return math.Sqrt(sum)
}

// unitVectorAt returns a unit vector pointing at vec1 from vec2. If the two
// positions are the same then a random unit vector is returned. We also return
// the distance between the points for use in the later height calculation.
func unitVectorAt(vec1 []float64, vec2 []float64) ([]float64, float64) {
	ret := diff(vec1, vec2)

	// If the coordinates aren't on top of each other we can normalize.
	if mag := magnitude(ret); mag > zeroThreshold {
		return mul(ret, 1.0/mag), mag
	}

	// Otherwise, just return a random unit vector.
	for i, _ := range ret {
		ret[i] = rand.Float64() - 0.5
	}
	if mag := magnitude(ret); mag > zeroThreshold {
		return mul(ret, 1.0/mag), 0.0
	}

	// And finally just give up and make a unit vector along the first
	// dimension. This should be exceedingly rare.
	ret = make([]float64, len(ret))
	ret[0] = 1.0
	return ret, 0.0
}
