// Package checker provides Docker specific implementations of the go-check.Checker interface.
package checker

import (
	"github.com/go-check/check"
	"github.com/vdemeester/shakers"
)

// As a commodity, we bring all check.Checker variables into the current namespace to avoid having
// to think about check.X versus checker.X.
var (
	DeepEquals   = check.DeepEquals
	ErrorMatches = check.ErrorMatches
	FitsTypeOf   = check.FitsTypeOf
	HasLen       = check.HasLen
	Implements   = check.Implements
	IsNil        = check.IsNil
	Matches      = check.Matches
	Not          = check.Not
	NotNil       = check.NotNil
	PanicMatches = check.PanicMatches
	Panics       = check.Panics

	Contains           = shakers.Contains
	ContainsAny        = shakers.ContainsAny
	Count              = shakers.Count
	Equals             = shakers.Equals
	EqualFold          = shakers.EqualFold
	False              = shakers.False
	GreaterOrEqualThan = shakers.GreaterOrEqualThan
	GreaterThan        = shakers.GreaterThan
	HasPrefix          = shakers.HasPrefix
	HasSuffix          = shakers.HasSuffix
	Index              = shakers.Index
	IndexAny           = shakers.IndexAny
	IsAfter            = shakers.IsAfter
	IsBefore           = shakers.IsBefore
	IsBetween          = shakers.IsBetween
	IsLower            = shakers.IsLower
	IsUpper            = shakers.IsUpper
	LessOrEqualThan    = shakers.LessOrEqualThan
	LessThan           = shakers.LessThan
	TimeEquals         = shakers.TimeEquals
	True               = shakers.True
	TimeIgnore         = shakers.TimeIgnore
)
