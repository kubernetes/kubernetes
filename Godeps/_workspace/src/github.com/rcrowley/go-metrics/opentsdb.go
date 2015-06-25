package metrics

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"strings"
	"time"
)

var shortHostName string = ""

// OpenTSDBConfig provides a container with configuration parameters for
// the OpenTSDB exporter
type OpenTSDBConfig struct {
	Addr          *net.TCPAddr  // Network address to connect to
	Registry      Registry      // Registry to be exported
	FlushInterval time.Duration // Flush interval
	DurationUnit  time.Duration // Time conversion unit for durations
	Prefix        string        // Prefix to be prepended to metric names
}

// OpenTSDB is a blocking exporter function which reports metrics in r
// to a TSDB server located at addr, flushing them every d duration
// and prepending metric names with prefix.
func OpenTSDB(r Registry, d time.Duration, prefix string, addr *net.TCPAddr) {
	OpenTSDBWithConfig(OpenTSDBConfig{
		Addr:          addr,
		Registry:      r,
		FlushInterval: d,
		DurationUnit:  time.Nanosecond,
		Prefix:        prefix,
	})
}

// OpenTSDBWithConfig is a blocking exporter function just like OpenTSDB,
// but it takes a OpenTSDBConfig instead.
func OpenTSDBWithConfig(c OpenTSDBConfig) {
	for _ = range time.Tick(c.FlushInterval) {
		if err := openTSDB(&c); nil != err {
			log.Println(err)
		}
	}
}

func getShortHostname() string {
	if shortHostName == "" {
		host, _ := os.Hostname()
		if index := strings.Index(host, "."); index > 0 {
			shortHostName = host[:index]
		} else {
			shortHostName = host
		}
	}
	return shortHostName
}

func openTSDB(c *OpenTSDBConfig) error {
	shortHostname := getShortHostname()
	now := time.Now().Unix()
	du := float64(c.DurationUnit)
	conn, err := net.DialTCP("tcp", nil, c.Addr)
	if nil != err {
		return err
	}
	defer conn.Close()
	w := bufio.NewWriter(conn)
	c.Registry.Each(func(name string, i interface{}) {
		switch metric := i.(type) {
		case Counter:
			fmt.Fprintf(w, "put %s.%s.count %d %d host=%s\n", c.Prefix, name, now, metric.Count(), shortHostname)
		case Gauge:
			fmt.Fprintf(w, "put %s.%s.value %d %d host=%s\n", c.Prefix, name, now, metric.Value(), shortHostname)
		case GaugeFloat64:
			fmt.Fprintf(w, "put %s.%s.value %d %f host=%s\n", c.Prefix, name, now, metric.Value(), shortHostname)
		case Histogram:
			h := metric.Snapshot()
			ps := h.Percentiles([]float64{0.5, 0.75, 0.95, 0.99, 0.999})
			fmt.Fprintf(w, "put %s.%s.count %d %d host=%s\n", c.Prefix, name, now, h.Count(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.min %d %d host=%s\n", c.Prefix, name, now, h.Min(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.max %d %d host=%s\n", c.Prefix, name, now, h.Max(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.mean %d %.2f host=%s\n", c.Prefix, name, now, h.Mean(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.std-dev %d %.2f host=%s\n", c.Prefix, name, now, h.StdDev(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.50-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[0], shortHostname)
			fmt.Fprintf(w, "put %s.%s.75-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[1], shortHostname)
			fmt.Fprintf(w, "put %s.%s.95-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[2], shortHostname)
			fmt.Fprintf(w, "put %s.%s.99-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[3], shortHostname)
			fmt.Fprintf(w, "put %s.%s.999-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[4], shortHostname)
		case Meter:
			m := metric.Snapshot()
			fmt.Fprintf(w, "put %s.%s.count %d %d host=%s\n", c.Prefix, name, now, m.Count(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.one-minute %d %.2f host=%s\n", c.Prefix, name, now, m.Rate1(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.five-minute %d %.2f host=%s\n", c.Prefix, name, now, m.Rate5(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.fifteen-minute %d %.2f host=%s\n", c.Prefix, name, now, m.Rate15(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.mean %d %.2f host=%s\n", c.Prefix, name, now, m.RateMean(), shortHostname)
		case Timer:
			t := metric.Snapshot()
			ps := t.Percentiles([]float64{0.5, 0.75, 0.95, 0.99, 0.999})
			fmt.Fprintf(w, "put %s.%s.count %d %d host=%s\n", c.Prefix, name, now, t.Count(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.min %d %d host=%s\n", c.Prefix, name, now, t.Min()/int64(du), shortHostname)
			fmt.Fprintf(w, "put %s.%s.max %d %d host=%s\n", c.Prefix, name, now, t.Max()/int64(du), shortHostname)
			fmt.Fprintf(w, "put %s.%s.mean %d %.2f host=%s\n", c.Prefix, name, now, t.Mean()/du, shortHostname)
			fmt.Fprintf(w, "put %s.%s.std-dev %d %.2f host=%s\n", c.Prefix, name, now, t.StdDev()/du, shortHostname)
			fmt.Fprintf(w, "put %s.%s.50-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[0]/du, shortHostname)
			fmt.Fprintf(w, "put %s.%s.75-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[1]/du, shortHostname)
			fmt.Fprintf(w, "put %s.%s.95-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[2]/du, shortHostname)
			fmt.Fprintf(w, "put %s.%s.99-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[3]/du, shortHostname)
			fmt.Fprintf(w, "put %s.%s.999-percentile %d %.2f host=%s\n", c.Prefix, name, now, ps[4]/du, shortHostname)
			fmt.Fprintf(w, "put %s.%s.one-minute %d %.2f host=%s\n", c.Prefix, name, now, t.Rate1(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.five-minute %d %.2f host=%s\n", c.Prefix, name, now, t.Rate5(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.fifteen-minute %d %.2f host=%s\n", c.Prefix, name, now, t.Rate15(), shortHostname)
			fmt.Fprintf(w, "put %s.%s.mean-rate %d %.2f host=%s\n", c.Prefix, name, now, t.RateMean(), shortHostname)
		}
		w.Flush()
	})
	return nil
}
