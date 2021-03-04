package leafnodes

import (
	"math"
	"time"

	"sync"

	"github.com/onsi/ginkgo/types"
)

type benchmarker struct {
	mu           sync.Mutex
	measurements map[string]*types.SpecMeasurement
	orderCounter int
}

func newBenchmarker() *benchmarker {
	return &benchmarker{
		measurements: make(map[string]*types.SpecMeasurement),
	}
}

func (b *benchmarker) Time(name string, body func(), info ...interface{}) (elapsedTime time.Duration) {
	t := time.Now()
	body()
	elapsedTime = time.Since(t)

	b.mu.Lock()
	defer b.mu.Unlock()
	measurement := b.getMeasurement(name, "Fastest Time", "Slowest Time", "Average Time", "s", 3, info...)
	measurement.Results = append(measurement.Results, elapsedTime.Seconds())

	return
}

func (b *benchmarker) RecordValue(name string, value float64, info ...interface{}) {
	b.mu.Lock()
	measurement := b.getMeasurement(name, "Smallest", " Largest", " Average", "", 3, info...)
	defer b.mu.Unlock()
	measurement.Results = append(measurement.Results, value)
}

func (b *benchmarker) RecordValueWithPrecision(name string, value float64, units string, precision int, info ...interface{}) {
	b.mu.Lock()
	measurement := b.getMeasurement(name, "Smallest", " Largest", " Average", units, precision, info...)
	defer b.mu.Unlock()
	measurement.Results = append(measurement.Results, value)
}

func (b *benchmarker) getMeasurement(name string, smallestLabel string, largestLabel string, averageLabel string, units string, precision int, info ...interface{}) *types.SpecMeasurement {
	measurement, ok := b.measurements[name]
	if !ok {
		var computedInfo interface{}
		computedInfo = nil
		if len(info) > 0 {
			computedInfo = info[0]
		}
		measurement = &types.SpecMeasurement{
			Name:          name,
			Info:          computedInfo,
			Order:         b.orderCounter,
			SmallestLabel: smallestLabel,
			LargestLabel:  largestLabel,
			AverageLabel:  averageLabel,
			Units:         units,
			Precision:     precision,
			Results:       make([]float64, 0),
		}
		b.measurements[name] = measurement
		b.orderCounter++
	}

	return measurement
}

func (b *benchmarker) measurementsReport() map[string]*types.SpecMeasurement {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, measurement := range b.measurements {
		measurement.Smallest = math.MaxFloat64
		measurement.Largest = -math.MaxFloat64
		sum := float64(0)
		sumOfSquares := float64(0)

		for _, result := range measurement.Results {
			if result > measurement.Largest {
				measurement.Largest = result
			}
			if result < measurement.Smallest {
				measurement.Smallest = result
			}
			sum += result
			sumOfSquares += result * result
		}

		n := float64(len(measurement.Results))
		measurement.Average = sum / n
		measurement.StdDeviation = math.Sqrt(sumOfSquares/n - (sum/n)*(sum/n))
	}

	return b.measurements
}
