package prometheus

import (
	"reflect"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
)

func TestGoCollector(t *testing.T) {
	var (
		c      = NewGoCollector()
		ch     = make(chan Metric)
		waitc  = make(chan struct{})
		closec = make(chan struct{})
		old    = -1
	)
	defer close(closec)

	go func() {
		c.Collect(ch)
		go func(c <-chan struct{}) {
			<-c
		}(closec)
		<-waitc
		c.Collect(ch)
	}()

	for {
		select {
		case metric := <-ch:
			switch m := metric.(type) {
			// Attention, this also catches Counter...
			case Gauge:
				pb := &dto.Metric{}
				m.Write(pb)

				if old == -1 {
					old = int(pb.GetGauge().GetValue())
					close(waitc)
					continue
				}

				if diff := int(pb.GetGauge().GetValue()) - old; diff != 1 {
					// TODO: This is flaky in highly concurrent situations.
					t.Errorf("want 1 new goroutine, got %d", diff)
				}

				return
			default:
				t.Errorf("want type Gauge, got %s", reflect.TypeOf(metric))
			}
		case <-time.After(1 * time.Second):
			t.Fatalf("expected collect timed out")
		}
	}
}
