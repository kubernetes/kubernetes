package view

import (
	"context"
	"testing"

	"go.opencensus.io/stats"
)

func TestMeasureFloat64AndInt64(t *testing.T) {
	// Recording through both a Float64Measure and Int64Measure with the
	// same name should work.

	im := stats.Int64("TestMeasureFloat64AndInt64", "", stats.UnitDimensionless)
	fm := stats.Float64("TestMeasureFloat64AndInt64", "", stats.UnitDimensionless)

	if im == nil || fm == nil {
		t.Fatal("Error creating Measures")
	}

	v1 := &View{
		Name:        "TestMeasureFloat64AndInt64/v1",
		Measure:     im,
		Aggregation: Sum(),
	}
	v2 := &View{
		Name:        "TestMeasureFloat64AndInt64/v2",
		Measure:     fm,
		Aggregation: Sum(),
	}
	Register(v1, v2)

	stats.Record(context.Background(), im.M(5))
	stats.Record(context.Background(), fm.M(2.2))

	d1, _ := RetrieveData(v1.Name)
	d2, _ := RetrieveData(v2.Name)

	sum1 := d1[0].Data.(*SumData)
	sum2 := d2[0].Data.(*SumData)

	// We expect both views to return 7.2, as though we recorded on a single measure.

	if got, want := sum1.Value, 7.2; got != want {
		t.Errorf("sum1 = %v; want %v", got, want)
	}
	if got, want := sum2.Value, 7.2; got != want {
		t.Errorf("sum2 = %v; want %v", got, want)
	}
}
