package metrics

import (
	"io/ioutil"
	"log"
	"sync"
	"testing"
)

const FANOUT = 128

// Stop the compiler from complaining during debugging.
var (
	_ = ioutil.Discard
	_ = log.LstdFlags
)

func BenchmarkMetrics(b *testing.B) {
	r := NewRegistry()
	c := NewRegisteredCounter("counter", r)
	g := NewRegisteredGauge("gauge", r)
	gf := NewRegisteredGaugeFloat64("gaugefloat64", r)
	h := NewRegisteredHistogram("histogram", r, NewUniformSample(100))
	m := NewRegisteredMeter("meter", r)
	t := NewRegisteredTimer("timer", r)
	RegisterDebugGCStats(r)
	RegisterRuntimeMemStats(r)
	b.ResetTimer()
	ch := make(chan bool)

	wgD := &sync.WaitGroup{}
	/*
		wgD.Add(1)
		go func() {
			defer wgD.Done()
			//log.Println("go CaptureDebugGCStats")
			for {
				select {
				case <-ch:
					//log.Println("done CaptureDebugGCStats")
					return
				default:
					CaptureDebugGCStatsOnce(r)
				}
			}
		}()
	//*/

	wgR := &sync.WaitGroup{}
	//*
	wgR.Add(1)
	go func() {
		defer wgR.Done()
		//log.Println("go CaptureRuntimeMemStats")
		for {
			select {
			case <-ch:
				//log.Println("done CaptureRuntimeMemStats")
				return
			default:
				CaptureRuntimeMemStatsOnce(r)
			}
		}
	}()
	//*/

	wgW := &sync.WaitGroup{}
	/*
		wgW.Add(1)
		go func() {
			defer wgW.Done()
			//log.Println("go Write")
			for {
				select {
				case <-ch:
					//log.Println("done Write")
					return
				default:
					WriteOnce(r, ioutil.Discard)
				}
			}
		}()
	//*/

	wg := &sync.WaitGroup{}
	wg.Add(FANOUT)
	for i := 0; i < FANOUT; i++ {
		go func(i int) {
			defer wg.Done()
			//log.Println("go", i)
			for i := 0; i < b.N; i++ {
				c.Inc(1)
				g.Update(int64(i))
				gf.Update(float64(i))
				h.Update(int64(i))
				m.Mark(1)
				t.Update(1)
			}
			//log.Println("done", i)
		}(i)
	}
	wg.Wait()
	close(ch)
	wgD.Wait()
	wgR.Wait()
	wgW.Wait()
}
