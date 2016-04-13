package main

import (
	"collectd.org/api"
	"collectd.org/network"

	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"
)

var nMeasurments = flag.Int("m", 1, "Number of measurements")
var tagVariance = flag.Int("v", 1, "Number of values per tag. Client is fixed at one tag")
var rate = flag.Int("r", 1, "Number of points per second")
var total = flag.Int("t", -1, "Total number of points to send (default is no limit)")
var host = flag.String("u", "127.0.0.1:25826", "Destination host in the form host:port")

func main() {
	flag.Parse()

	conn, err := network.Dial(*host, network.ClientOptions{})
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer conn.Close()

	rateLimiter := make(chan int, *rate)

	go func() {
		ticker := time.NewTicker(time.Second)
		for {
			select {
			case <-ticker.C:
				for i := 0; i < *rate; i++ {
					rateLimiter <- i
				}
			}
		}
	}()

	nSent := 0
	for {
		if nSent >= *total && *total > 0 {
			break
		}
		<-rateLimiter

		vl := api.ValueList{
			Identifier: api.Identifier{
				Host:   "tagvalue" + strconv.Itoa(int(rand.Int31n(int32(*tagVariance)))),
				Plugin: "golang" + strconv.Itoa(int(rand.Int31n(int32(*nMeasurments)))),
				Type:   "gauge",
			},
			Time:     time.Now(),
			Interval: 10 * time.Second,
			Values:   []api.Value{api.Gauge(42.0)},
		}
		if err := conn.Write(vl); err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		conn.Flush()
		nSent = nSent + 1
	}

	fmt.Println("Number of points sent:", nSent)
}
