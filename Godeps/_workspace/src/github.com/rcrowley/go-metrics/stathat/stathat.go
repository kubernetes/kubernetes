// Metrics output to StatHat.
package stathat

import (
	"github.com/rcrowley/go-metrics"
	"github.com/stathat/go"
	"log"
	"time"
)

func Stathat(r metrics.Registry, d time.Duration, userkey string) {
	for {
		if err := sh(r, userkey); nil != err {
			log.Println(err)
		}
		time.Sleep(d)
	}
}

func sh(r metrics.Registry, userkey string) error {
	r.Each(func(name string, i interface{}) {
		switch metric := i.(type) {
		case metrics.Counter:
			stathat.PostEZCount(name, userkey, int(metric.Count()))
		case metrics.Gauge:
			stathat.PostEZValue(name, userkey, float64(metric.Value()))
		case metrics.GaugeFloat64:
			stathat.PostEZValue(name, userkey, float64(metric.Value()))
		case metrics.Histogram:
			h := metric.Snapshot()
			ps := h.Percentiles([]float64{0.5, 0.75, 0.95, 0.99, 0.999})
			stathat.PostEZCount(name+".count", userkey, int(h.Count()))
			stathat.PostEZValue(name+".min", userkey, float64(h.Min()))
			stathat.PostEZValue(name+".max", userkey, float64(h.Max()))
			stathat.PostEZValue(name+".mean", userkey, float64(h.Mean()))
			stathat.PostEZValue(name+".std-dev", userkey, float64(h.StdDev()))
			stathat.PostEZValue(name+".50-percentile", userkey, float64(ps[0]))
			stathat.PostEZValue(name+".75-percentile", userkey, float64(ps[1]))
			stathat.PostEZValue(name+".95-percentile", userkey, float64(ps[2]))
			stathat.PostEZValue(name+".99-percentile", userkey, float64(ps[3]))
			stathat.PostEZValue(name+".999-percentile", userkey, float64(ps[4]))
		case metrics.Meter:
			m := metric.Snapshot()
			stathat.PostEZCount(name+".count", userkey, int(m.Count()))
			stathat.PostEZValue(name+".one-minute", userkey, float64(m.Rate1()))
			stathat.PostEZValue(name+".five-minute", userkey, float64(m.Rate5()))
			stathat.PostEZValue(name+".fifteen-minute", userkey, float64(m.Rate15()))
			stathat.PostEZValue(name+".mean", userkey, float64(m.RateMean()))
		case metrics.Timer:
			t := metric.Snapshot()
			ps := t.Percentiles([]float64{0.5, 0.75, 0.95, 0.99, 0.999})
			stathat.PostEZCount(name+".count", userkey, int(t.Count()))
			stathat.PostEZValue(name+".min", userkey, float64(t.Min()))
			stathat.PostEZValue(name+".max", userkey, float64(t.Max()))
			stathat.PostEZValue(name+".mean", userkey, float64(t.Mean()))
			stathat.PostEZValue(name+".std-dev", userkey, float64(t.StdDev()))
			stathat.PostEZValue(name+".50-percentile", userkey, float64(ps[0]))
			stathat.PostEZValue(name+".75-percentile", userkey, float64(ps[1]))
			stathat.PostEZValue(name+".95-percentile", userkey, float64(ps[2]))
			stathat.PostEZValue(name+".99-percentile", userkey, float64(ps[3]))
			stathat.PostEZValue(name+".999-percentile", userkey, float64(ps[4]))
			stathat.PostEZValue(name+".one-minute", userkey, float64(t.Rate1()))
			stathat.PostEZValue(name+".five-minute", userkey, float64(t.Rate5()))
			stathat.PostEZValue(name+".fifteen-minute", userkey, float64(t.Rate15()))
			stathat.PostEZValue(name+".mean-rate", userkey, float64(t.RateMean()))
		}
	})
	return nil
}
