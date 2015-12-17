package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"time"
)

func main() {
	var (
		intervalN = flag.Int("interval", 10, "interval")
		seriesN   = flag.Int("series", 1, "Number of unique series to generate.")
		clientN   = flag.Int("clients", 10, "Number of clients to simulate.")
		host      = flag.String("host", "localhost:8086", "Target host")
	)
	flag.Parse()

	// Calculate time so that the last point ends now.
	n := (*clientN) * (*seriesN) * (*intervalN)
	t := time.Now().UTC().Add(-time.Duration(n) * time.Second)

	for i := 0; i < *clientN; i++ {
		for j := 0; j < *seriesN; j++ {
			points := make([]*Point, 0)
			for k := 0; k < *intervalN; k++ {
				t = t.Add(1 * time.Second)
				points = append(points, &Point{
					Measurement: "cpu",
					Time:        t,
					Tags:        map[string]string{"host": fmt.Sprintf("server%d", j+1)},
					Fields:      map[string]interface{}{"value": 100},
				})
			}
			batch := &Batch{
				Database:        "db",
				RetentionPolicy: "raw",
				Points:          points,
			}
			buf, _ := json.Marshal(batch)

			fmt.Printf("http://%s/write POST %s\n", *host, buf)
		}
	}
}

type Batch struct {
	Database        string   `json:"database"`
	RetentionPolicy string   `json:"retentionPolicy"`
	Points          []*Point `json:"points"`
}

type Point struct {
	Measurement string                 `json:"measurement"`
	Time        time.Time              `json:"time"`
	Tags        map[string]string      `json:"tags"`
	Fields      map[string]interface{} `json:"fields"`
}
