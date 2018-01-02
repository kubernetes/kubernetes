package inf_test

import (
	"math/big"
	"testing"

	"gopkg.in/inf.v0"
)

var decRounderInputs = [...]struct {
	quo    *inf.Dec
	rA, rB *big.Int
}{
	// examples from go language spec
	{inf.NewDec(1, 0), big.NewInt(2), big.NewInt(3)},   //  5 /  3
	{inf.NewDec(-1, 0), big.NewInt(-2), big.NewInt(3)}, // -5 /  3
	{inf.NewDec(-1, 0), big.NewInt(2), big.NewInt(-3)}, //  5 / -3
	{inf.NewDec(1, 0), big.NewInt(-2), big.NewInt(-3)}, // -5 / -3
	// examples from godoc
	{inf.NewDec(-1, 1), big.NewInt(-8), big.NewInt(10)},
	{inf.NewDec(-1, 1), big.NewInt(-5), big.NewInt(10)},
	{inf.NewDec(-1, 1), big.NewInt(-2), big.NewInt(10)},
	{inf.NewDec(0, 1), big.NewInt(-8), big.NewInt(10)},
	{inf.NewDec(0, 1), big.NewInt(-5), big.NewInt(10)},
	{inf.NewDec(0, 1), big.NewInt(-2), big.NewInt(10)},
	{inf.NewDec(0, 1), big.NewInt(0), big.NewInt(1)},
	{inf.NewDec(0, 1), big.NewInt(2), big.NewInt(10)},
	{inf.NewDec(0, 1), big.NewInt(5), big.NewInt(10)},
	{inf.NewDec(0, 1), big.NewInt(8), big.NewInt(10)},
	{inf.NewDec(1, 1), big.NewInt(2), big.NewInt(10)},
	{inf.NewDec(1, 1), big.NewInt(5), big.NewInt(10)},
	{inf.NewDec(1, 1), big.NewInt(8), big.NewInt(10)},
}

var decRounderResults = [...]struct {
	rounder inf.Rounder
	results [len(decRounderInputs)]*inf.Dec
}{
	{inf.RoundExact, [...]*inf.Dec{nil, nil, nil, nil,
		nil, nil, nil, nil, nil, nil,
		inf.NewDec(0, 1), nil, nil, nil, nil, nil, nil}},
	{inf.RoundDown, [...]*inf.Dec{
		inf.NewDec(1, 0), inf.NewDec(-1, 0), inf.NewDec(-1, 0), inf.NewDec(1, 0),
		inf.NewDec(-1, 1), inf.NewDec(-1, 1), inf.NewDec(-1, 1),
		inf.NewDec(0, 1), inf.NewDec(0, 1), inf.NewDec(0, 1),
		inf.NewDec(0, 1),
		inf.NewDec(0, 1), inf.NewDec(0, 1), inf.NewDec(0, 1),
		inf.NewDec(1, 1), inf.NewDec(1, 1), inf.NewDec(1, 1)}},
	{inf.RoundUp, [...]*inf.Dec{
		inf.NewDec(2, 0), inf.NewDec(-2, 0), inf.NewDec(-2, 0), inf.NewDec(2, 0),
		inf.NewDec(-2, 1), inf.NewDec(-2, 1), inf.NewDec(-2, 1),
		inf.NewDec(-1, 1), inf.NewDec(-1, 1), inf.NewDec(-1, 1),
		inf.NewDec(0, 1),
		inf.NewDec(1, 1), inf.NewDec(1, 1), inf.NewDec(1, 1),
		inf.NewDec(2, 1), inf.NewDec(2, 1), inf.NewDec(2, 1)}},
	{inf.RoundHalfDown, [...]*inf.Dec{
		inf.NewDec(2, 0), inf.NewDec(-2, 0), inf.NewDec(-2, 0), inf.NewDec(2, 0),
		inf.NewDec(-2, 1), inf.NewDec(-1, 1), inf.NewDec(-1, 1),
		inf.NewDec(-1, 1), inf.NewDec(0, 1), inf.NewDec(0, 1),
		inf.NewDec(0, 1),
		inf.NewDec(0, 1), inf.NewDec(0, 1), inf.NewDec(1, 1),
		inf.NewDec(1, 1), inf.NewDec(1, 1), inf.NewDec(2, 1)}},
	{inf.RoundHalfUp, [...]*inf.Dec{
		inf.NewDec(2, 0), inf.NewDec(-2, 0), inf.NewDec(-2, 0), inf.NewDec(2, 0),
		inf.NewDec(-2, 1), inf.NewDec(-2, 1), inf.NewDec(-1, 1),
		inf.NewDec(-1, 1), inf.NewDec(-1, 1), inf.NewDec(0, 1),
		inf.NewDec(0, 1),
		inf.NewDec(0, 1), inf.NewDec(1, 1), inf.NewDec(1, 1),
		inf.NewDec(1, 1), inf.NewDec(2, 1), inf.NewDec(2, 1)}},
	{inf.RoundHalfEven, [...]*inf.Dec{
		inf.NewDec(2, 0), inf.NewDec(-2, 0), inf.NewDec(-2, 0), inf.NewDec(2, 0),
		inf.NewDec(-2, 1), inf.NewDec(-2, 1), inf.NewDec(-1, 1),
		inf.NewDec(-1, 1), inf.NewDec(0, 1), inf.NewDec(0, 1),
		inf.NewDec(0, 1),
		inf.NewDec(0, 1), inf.NewDec(0, 1), inf.NewDec(1, 1),
		inf.NewDec(1, 1), inf.NewDec(2, 1), inf.NewDec(2, 1)}},
	{inf.RoundFloor, [...]*inf.Dec{
		inf.NewDec(1, 0), inf.NewDec(-2, 0), inf.NewDec(-2, 0), inf.NewDec(1, 0),
		inf.NewDec(-2, 1), inf.NewDec(-2, 1), inf.NewDec(-2, 1),
		inf.NewDec(-1, 1), inf.NewDec(-1, 1), inf.NewDec(-1, 1),
		inf.NewDec(0, 1),
		inf.NewDec(0, 1), inf.NewDec(0, 1), inf.NewDec(0, 1),
		inf.NewDec(1, 1), inf.NewDec(1, 1), inf.NewDec(1, 1)}},
	{inf.RoundCeil, [...]*inf.Dec{
		inf.NewDec(2, 0), inf.NewDec(-1, 0), inf.NewDec(-1, 0), inf.NewDec(2, 0),
		inf.NewDec(-1, 1), inf.NewDec(-1, 1), inf.NewDec(-1, 1),
		inf.NewDec(0, 1), inf.NewDec(0, 1), inf.NewDec(0, 1),
		inf.NewDec(0, 1),
		inf.NewDec(1, 1), inf.NewDec(1, 1), inf.NewDec(1, 1),
		inf.NewDec(2, 1), inf.NewDec(2, 1), inf.NewDec(2, 1)}},
}

func TestDecRounders(t *testing.T) {
	for i, a := range decRounderResults {
		for j, input := range decRounderInputs {
			q := new(inf.Dec).Set(input.quo)
			rA, rB := new(big.Int).Set(input.rA), new(big.Int).Set(input.rB)
			res := a.rounder.Round(new(inf.Dec), q, rA, rB)
			if a.results[j] == nil && res == nil {
				continue
			}
			if (a.results[j] == nil && res != nil) ||
				(a.results[j] != nil && res == nil) ||
				a.results[j].Cmp(res) != 0 {
				t.Errorf("#%d,%d Rounder got %v; expected %v", i, j, res, a.results[j])
			}
		}
	}
}
