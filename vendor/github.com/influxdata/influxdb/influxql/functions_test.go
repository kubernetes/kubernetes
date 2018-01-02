package influxql_test

import (
	"math"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/pkg/deep"
)

func almostEqual(got, exp float64) bool {
	return math.Abs(got-exp) < 1e-5 && !math.IsNaN(got)
}

func TestHoltWinters_AusTourists(t *testing.T) {
	hw := influxql.NewFloatHoltWintersReducer(10, 4, false, 1)
	// Dataset from http://www.inside-r.org/packages/cran/fpp/docs/austourists
	austourists := []influxql.FloatPoint{
		{Time: 1, Value: 30.052513},
		{Time: 2, Value: 19.148496},
		{Time: 3, Value: 25.317692},
		{Time: 4, Value: 27.591437},
		{Time: 5, Value: 32.076456},
		{Time: 6, Value: 23.487961},
		{Time: 7, Value: 28.47594},
		{Time: 8, Value: 35.123753},
		{Time: 9, Value: 36.838485},
		{Time: 10, Value: 25.007017},
		{Time: 11, Value: 30.72223},
		{Time: 12, Value: 28.693759},
		{Time: 13, Value: 36.640986},
		{Time: 14, Value: 23.824609},
		{Time: 15, Value: 29.311683},
		{Time: 16, Value: 31.770309},
		{Time: 17, Value: 35.177877},
		{Time: 18, Value: 19.775244},
		{Time: 19, Value: 29.60175},
		{Time: 20, Value: 34.538842},
		{Time: 21, Value: 41.273599},
		{Time: 22, Value: 26.655862},
		{Time: 23, Value: 28.279859},
		{Time: 24, Value: 35.191153},
		{Time: 25, Value: 41.727458},
		{Time: 26, Value: 24.04185},
		{Time: 27, Value: 32.328103},
		{Time: 28, Value: 37.328708},
		{Time: 29, Value: 46.213153},
		{Time: 30, Value: 29.346326},
		{Time: 31, Value: 36.48291},
		{Time: 32, Value: 42.977719},
		{Time: 33, Value: 48.901525},
		{Time: 34, Value: 31.180221},
		{Time: 35, Value: 37.717881},
		{Time: 36, Value: 40.420211},
		{Time: 37, Value: 51.206863},
		{Time: 38, Value: 31.887228},
		{Time: 39, Value: 40.978263},
		{Time: 40, Value: 43.772491},
		{Time: 41, Value: 55.558567},
		{Time: 42, Value: 33.850915},
		{Time: 43, Value: 42.076383},
		{Time: 44, Value: 45.642292},
		{Time: 45, Value: 59.76678},
		{Time: 46, Value: 35.191877},
		{Time: 47, Value: 44.319737},
		{Time: 48, Value: 47.913736},
	}

	for _, p := range austourists {
		hw.AggregateFloat(&p)
	}
	points := hw.Emit()

	forecasted := []influxql.FloatPoint{
		{Time: 49, Value: 51.85064132137853},
		{Time: 50, Value: 43.26055282315273},
		{Time: 51, Value: 41.827258044814464},
		{Time: 52, Value: 54.3990354591749},
		{Time: 53, Value: 54.62334472770803},
		{Time: 54, Value: 45.57155693625209},
		{Time: 55, Value: 44.06051240252263},
		{Time: 56, Value: 57.30029870759433},
		{Time: 57, Value: 57.53591513519172},
		{Time: 58, Value: 47.999008139396096},
	}

	if exp, got := len(forecasted), len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}

	for i := range forecasted {
		if exp, got := forecasted[i].Time, points[i].Time; got != exp {
			t.Errorf("unexpected time on points[%d] got %v exp %v", i, got, exp)
		}
		if exp, got := forecasted[i].Value, points[i].Value; !almostEqual(got, exp) {
			t.Errorf("unexpected value on points[%d] got %v exp %v", i, got, exp)
		}
	}
}

func TestHoltWinters_AusTourists_Missing(t *testing.T) {
	hw := influxql.NewFloatHoltWintersReducer(10, 4, false, 1)
	// Dataset from http://www.inside-r.org/packages/cran/fpp/docs/austourists
	austourists := []influxql.FloatPoint{
		{Time: 1, Value: 30.052513},
		{Time: 3, Value: 25.317692},
		{Time: 4, Value: 27.591437},
		{Time: 5, Value: 32.076456},
		{Time: 6, Value: 23.487961},
		{Time: 7, Value: 28.47594},
		{Time: 9, Value: 36.838485},
		{Time: 10, Value: 25.007017},
		{Time: 11, Value: 30.72223},
		{Time: 12, Value: 28.693759},
		{Time: 13, Value: 36.640986},
		{Time: 14, Value: 23.824609},
		{Time: 15, Value: 29.311683},
		{Time: 16, Value: 31.770309},
		{Time: 17, Value: 35.177877},
		{Time: 19, Value: 29.60175},
		{Time: 20, Value: 34.538842},
		{Time: 21, Value: 41.273599},
		{Time: 22, Value: 26.655862},
		{Time: 23, Value: 28.279859},
		{Time: 24, Value: 35.191153},
		{Time: 25, Value: 41.727458},
		{Time: 26, Value: 24.04185},
		{Time: 27, Value: 32.328103},
		{Time: 28, Value: 37.328708},
		{Time: 30, Value: 29.346326},
		{Time: 31, Value: 36.48291},
		{Time: 32, Value: 42.977719},
		{Time: 34, Value: 31.180221},
		{Time: 35, Value: 37.717881},
		{Time: 36, Value: 40.420211},
		{Time: 37, Value: 51.206863},
		{Time: 38, Value: 31.887228},
		{Time: 41, Value: 55.558567},
		{Time: 42, Value: 33.850915},
		{Time: 43, Value: 42.076383},
		{Time: 44, Value: 45.642292},
		{Time: 45, Value: 59.76678},
		{Time: 46, Value: 35.191877},
		{Time: 47, Value: 44.319737},
		{Time: 48, Value: 47.913736},
	}

	for _, p := range austourists {
		hw.AggregateFloat(&p)
	}
	points := hw.Emit()

	forecasted := []influxql.FloatPoint{
		{Time: 49, Value: 54.84533610387743},
		{Time: 50, Value: 41.19329421863249},
		{Time: 51, Value: 45.71673175112451},
		{Time: 52, Value: 56.05759298805955},
		{Time: 53, Value: 59.32337460282217},
		{Time: 54, Value: 44.75280096850461},
		{Time: 55, Value: 49.98865098113751},
		{Time: 56, Value: 61.86084934967605},
		{Time: 57, Value: 65.95805633454883},
		{Time: 58, Value: 50.1502170480547},
	}

	if exp, got := len(forecasted), len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}

	for i := range forecasted {
		if exp, got := forecasted[i].Time, points[i].Time; got != exp {
			t.Errorf("unexpected time on points[%d] got %v exp %v", i, got, exp)
		}
		if exp, got := forecasted[i].Value, points[i].Value; !almostEqual(got, exp) {
			t.Errorf("unexpected value on points[%d] got %v exp %v", i, got, exp)
		}
	}
}

func TestHoltWinters_USPopulation(t *testing.T) {
	series := []influxql.FloatPoint{
		{Time: 1, Value: 3.93},
		{Time: 2, Value: 5.31},
		{Time: 3, Value: 7.24},
		{Time: 4, Value: 9.64},
		{Time: 5, Value: 12.90},
		{Time: 6, Value: 17.10},
		{Time: 7, Value: 23.20},
		{Time: 8, Value: 31.40},
		{Time: 9, Value: 39.80},
		{Time: 10, Value: 50.20},
		{Time: 11, Value: 62.90},
		{Time: 12, Value: 76.00},
		{Time: 13, Value: 92.00},
		{Time: 14, Value: 105.70},
		{Time: 15, Value: 122.80},
		{Time: 16, Value: 131.70},
		{Time: 17, Value: 151.30},
		{Time: 18, Value: 179.30},
		{Time: 19, Value: 203.20},
	}
	hw := influxql.NewFloatHoltWintersReducer(10, 0, true, 1)
	for _, p := range series {
		hw.AggregateFloat(&p)
	}
	points := hw.Emit()

	forecasted := []influxql.FloatPoint{
		{Time: 1, Value: 3.93},
		{Time: 2, Value: 4.957405463559748},
		{Time: 3, Value: 7.012210102535647},
		{Time: 4, Value: 10.099589257439924},
		{Time: 5, Value: 14.229926188104242},
		{Time: 6, Value: 19.418878968703797},
		{Time: 7, Value: 25.68749172281409},
		{Time: 8, Value: 33.062351305731305},
		{Time: 9, Value: 41.575791076125206},
		{Time: 10, Value: 51.26614395589263},
		{Time: 11, Value: 62.178047564264595},
		{Time: 12, Value: 74.36280483872488},
		{Time: 13, Value: 87.87880423073163},
		{Time: 14, Value: 102.79200429905801},
		{Time: 15, Value: 119.17648832929542},
		{Time: 16, Value: 137.11509549747296},
		{Time: 17, Value: 156.70013608313175},
		{Time: 18, Value: 178.03419933863566},
		{Time: 19, Value: 201.23106385518594},
		{Time: 20, Value: 226.4167216525905},
		{Time: 21, Value: 253.73052878285205},
		{Time: 22, Value: 283.32649700397553},
		{Time: 23, Value: 315.37474308085984},
		{Time: 24, Value: 350.06311454009256},
		{Time: 25, Value: 387.59901328556873},
		{Time: 26, Value: 428.21144141893404},
		{Time: 27, Value: 472.1532969569147},
		{Time: 28, Value: 519.7039509590035},
		{Time: 29, Value: 571.1721419458248},
	}

	if exp, got := len(forecasted), len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}
	for i := range forecasted {
		if exp, got := forecasted[i].Time, points[i].Time; got != exp {
			t.Errorf("unexpected time on points[%d] got %v exp %v", i, got, exp)
		}
		if exp, got := forecasted[i].Value, points[i].Value; !almostEqual(got, exp) {
			t.Errorf("unexpected value on points[%d] got %v exp %v", i, got, exp)
		}
	}
}

func TestHoltWinters_USPopulation_Missing(t *testing.T) {
	series := []influxql.FloatPoint{
		{Time: 1, Value: 3.93},
		{Time: 2, Value: 5.31},
		{Time: 3, Value: 7.24},
		{Time: 4, Value: 9.64},
		{Time: 5, Value: 12.90},
		{Time: 6, Value: 17.10},
		{Time: 7, Value: 23.20},
		{Time: 8, Value: 31.40},
		{Time: 10, Value: 50.20},
		{Time: 11, Value: 62.90},
		{Time: 12, Value: 76.00},
		{Time: 13, Value: 92.00},
		{Time: 15, Value: 122.80},
		{Time: 16, Value: 131.70},
		{Time: 17, Value: 151.30},
		{Time: 19, Value: 203.20},
	}
	hw := influxql.NewFloatHoltWintersReducer(10, 0, true, 1)
	for _, p := range series {
		hw.AggregateFloat(&p)
	}
	points := hw.Emit()

	forecasted := []influxql.FloatPoint{
		{Time: 1, Value: 3.93},
		{Time: 2, Value: 4.8931364428135105},
		{Time: 3, Value: 6.962653629047061},
		{Time: 4, Value: 10.056207765903274},
		{Time: 5, Value: 14.18435088129532},
		{Time: 6, Value: 19.362939306110846},
		{Time: 7, Value: 25.613247940326584},
		{Time: 8, Value: 32.96213087008264},
		{Time: 9, Value: 41.442230043017204},
		{Time: 10, Value: 51.09223428526052},
		{Time: 11, Value: 61.95719155158485},
		{Time: 12, Value: 74.08887794968567},
		{Time: 13, Value: 87.54622778052787},
		{Time: 14, Value: 102.39582960014131},
		{Time: 15, Value: 118.7124941463221},
		{Time: 16, Value: 136.57990089987464},
		{Time: 17, Value: 156.09133107941278},
		{Time: 18, Value: 177.35049601833734},
		{Time: 19, Value: 200.472471161683},
		{Time: 20, Value: 225.58474737097785},
		{Time: 21, Value: 252.82841286206823},
		{Time: 22, Value: 282.35948095261017},
		{Time: 23, Value: 314.3503808953992},
		{Time: 24, Value: 348.99163145856954},
		{Time: 25, Value: 386.49371962730555},
		{Time: 26, Value: 427.08920989407727},
		{Time: 27, Value: 471.0351131332573},
		{Time: 28, Value: 518.615548088049},
		{Time: 29, Value: 570.1447331101863},
	}

	if exp, got := len(forecasted), len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}
	for i := range forecasted {
		if exp, got := forecasted[i].Time, points[i].Time; got != exp {
			t.Errorf("unexpected time on points[%d] got %v exp %v", i, got, exp)
		}
		if exp, got := forecasted[i].Value, points[i].Value; !almostEqual(got, exp) {
			t.Errorf("unexpected value on points[%d] got %v exp %v", i, got, exp)
		}
	}
}
func TestHoltWinters_RoundTime(t *testing.T) {
	maxTime := time.Unix(0, influxql.MaxTime).Round(time.Second).UnixNano()
	data := []influxql.FloatPoint{
		{Time: maxTime - int64(5*time.Second), Value: 1},
		{Time: maxTime - int64(4*time.Second+103*time.Millisecond), Value: 10},
		{Time: maxTime - int64(3*time.Second+223*time.Millisecond), Value: 2},
		{Time: maxTime - int64(2*time.Second+481*time.Millisecond), Value: 11},
	}
	hw := influxql.NewFloatHoltWintersReducer(2, 2, true, time.Second)
	for _, p := range data {
		hw.AggregateFloat(&p)
	}
	points := hw.Emit()

	forecasted := []influxql.FloatPoint{
		{Time: maxTime - int64(5*time.Second), Value: 1},
		{Time: maxTime - int64(4*time.Second), Value: 10.006729104838234},
		{Time: maxTime - int64(3*time.Second), Value: 1.998341814469269},
		{Time: maxTime - int64(2*time.Second), Value: 10.997858830631172},
		{Time: maxTime - int64(1*time.Second), Value: 4.085860238030013},
		{Time: maxTime - int64(0*time.Second), Value: 11.35713604403339},
	}

	if exp, got := len(forecasted), len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}
	for i := range forecasted {
		if exp, got := forecasted[i].Time, points[i].Time; got != exp {
			t.Errorf("unexpected time on points[%d] got %v exp %v", i, got, exp)
		}
		if exp, got := forecasted[i].Value, points[i].Value; !almostEqual(got, exp) {
			t.Errorf("unexpected value on points[%d] got %v exp %v", i, got, exp)
		}
	}
}

func TestHoltWinters_MaxTime(t *testing.T) {
	data := []influxql.FloatPoint{
		{Time: influxql.MaxTime - 1, Value: 1},
		{Time: influxql.MaxTime, Value: 2},
	}
	hw := influxql.NewFloatHoltWintersReducer(1, 0, true, 1)
	for _, p := range data {
		hw.AggregateFloat(&p)
	}
	points := hw.Emit()

	forecasted := []influxql.FloatPoint{
		{Time: influxql.MaxTime - 1, Value: 1},
		{Time: influxql.MaxTime, Value: 2.001516944066403},
		{Time: influxql.MaxTime + 1, Value: 2.5365248972488343},
	}

	if exp, got := len(forecasted), len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}
	for i := range forecasted {
		if exp, got := forecasted[i].Time, points[i].Time; got != exp {
			t.Errorf("unexpected time on points[%d] got %v exp %v", i, got, exp)
		}
		if exp, got := forecasted[i].Value, points[i].Value; !almostEqual(got, exp) {
			t.Errorf("unexpected value on points[%d] got %v exp %v", i, got, exp)
		}
	}
}

// TestSample_AllSamplesSeen attempts to verify that it is possible
// to get every subsample in a reasonable number of iterations.
//
// The idea here is that 6 iterations should be enough to hit every possible
// sequence atleast once.
func TestSample_AllSamplesSeen(t *testing.T) {

	ps := []influxql.FloatPoint{
		{Time: 1, Value: 1},
		{Time: 2, Value: 2},
		{Time: 3, Value: 3},
	}

	// List of all the possible subsamples
	samples := [][]influxql.FloatPoint{
		{
			{Time: 1, Value: 1},
			{Time: 2, Value: 2},
		},
		{
			{Time: 1, Value: 1},
			{Time: 3, Value: 3},
		},
		{
			{Time: 2, Value: 2},
			{Time: 3, Value: 3},
		},
	}

	// 6 iterations should be more than sufficient to garentee that
	// we hit every possible subsample.
	for i := 0; i < 6; i++ {
		s := influxql.NewFloatSampleReducer(2)
		for _, p := range ps {
			s.AggregateFloat(&p)
		}

		points := s.Emit()

		// if samples is empty we've seen every sample, so we're done
		if len(samples) == 0 {
			return
		}

		for i, sample := range samples {
			// if we find a sample that it matches, remove it from
			// this list of possible samples
			if deep.Equal(sample, points) {
				samples = append(samples[:i], samples[i+1:]...)
			}
		}

	}

	// If we missed a sample, report the error
	if exp, got := 0, len(samples); exp != got {
		t.Fatalf("expected to get every sample: got %d, exp %d", got, exp)
	}

}

func TestSample_SampleSizeLessThanNumPoints(t *testing.T) {
	s := influxql.NewFloatSampleReducer(2)

	ps := []influxql.FloatPoint{
		{Time: 1, Value: 1},
		{Time: 2, Value: 2},
		{Time: 3, Value: 3},
	}

	for _, p := range ps {
		s.AggregateFloat(&p)
	}

	points := s.Emit()

	if exp, got := 2, len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}
}

func TestSample_SampleSizeGreaterThanNumPoints(t *testing.T) {
	s := influxql.NewFloatSampleReducer(4)

	ps := []influxql.FloatPoint{
		{Time: 1, Value: 1},
		{Time: 2, Value: 2},
		{Time: 3, Value: 3},
	}

	for _, p := range ps {
		s.AggregateFloat(&p)
	}

	points := s.Emit()

	if exp, got := len(ps), len(points); exp != got {
		t.Fatalf("unexpected number of points emitted: got %d exp %d", got, exp)
	}

	if !deep.Equal(ps, points) {
		t.Fatalf("unexpected points: %s", spew.Sdump(points))
	}
}
