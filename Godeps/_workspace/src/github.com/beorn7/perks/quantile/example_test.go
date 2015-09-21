// +build go1.1

package quantile_test

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/beorn7/perks/quantile"
)

func Example_simple() {
	ch := make(chan float64)
	go sendFloats(ch)

	// Compute the 50th, 90th, and 99th percentile.
	q := quantile.NewTargeted(map[float64]float64{
		0.50: 0.005,
		0.90: 0.001,
		0.99: 0.0001,
	})
	for v := range ch {
		q.Insert(v)
	}

	fmt.Println("perc50:", q.Query(0.50))
	fmt.Println("perc90:", q.Query(0.90))
	fmt.Println("perc99:", q.Query(0.99))
	fmt.Println("count:", q.Count())
	// Output:
	// perc50: 5
	// perc90: 16
	// perc99: 223
	// count: 2388
}

func Example_mergeMultipleStreams() {
	// Scenario:
	// We have multiple database shards. On each shard, there is a process
	// collecting query response times from the database logs and inserting
	// them into a Stream (created via NewTargeted(0.90)), much like the
	// Simple example. These processes expose a network interface for us to
	// ask them to serialize and send us the results of their
	// Stream.Samples so we may Merge and Query them.
	//
	// NOTES:
	// * These sample sets are small, allowing us to get them
	// across the network much faster than sending the entire list of data
	// points.
	//
	// * For this to work correctly, we must supply the same quantiles
	// a priori the process collecting the samples supplied to NewTargeted,
	// even if we do not plan to query them all here.
	ch := make(chan quantile.Samples)
	getDBQuerySamples(ch)
	q := quantile.NewTargeted(map[float64]float64{0.90: 0.001})
	for samples := range ch {
		q.Merge(samples)
	}
	fmt.Println("perc90:", q.Query(0.90))
}

func Example_window() {
	// Scenario: We want the 90th, 95th, and 99th percentiles for each
	// minute.

	ch := make(chan float64)
	go sendStreamValues(ch)

	tick := time.NewTicker(1 * time.Minute)
	q := quantile.NewTargeted(map[float64]float64{
		0.90: 0.001,
		0.95: 0.0005,
		0.99: 0.0001,
	})
	for {
		select {
		case t := <-tick.C:
			flushToDB(t, q.Samples())
			q.Reset()
		case v := <-ch:
			q.Insert(v)
		}
	}
}

func sendStreamValues(ch chan float64) {
	// Use your imagination
}

func flushToDB(t time.Time, samples quantile.Samples) {
	// Use your imagination
}

// This is a stub for the above example. In reality this would hit the remote
// servers via http or something like it.
func getDBQuerySamples(ch chan quantile.Samples) {}

func sendFloats(ch chan<- float64) {
	f, err := os.Open("exampledata.txt")
	if err != nil {
		log.Fatal(err)
	}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		b := sc.Bytes()
		v, err := strconv.ParseFloat(string(b), 64)
		if err != nil {
			log.Fatal(err)
		}
		ch <- v
	}
	if sc.Err() != nil {
		log.Fatal(sc.Err())
	}
	close(ch)
}
