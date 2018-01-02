package client_test

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/influxdata/influxdb/client/v2"
)

// Create a new client
func ExampleClient() {
	// NOTE: this assumes you've setup a user and have setup shell env variables,
	// namely INFLUX_USER/INFLUX_PWD. If not just omit Username/Password below.
	_, err := client.NewHTTPClient(client.HTTPConfig{
		Addr:     "http://localhost:8086",
		Username: os.Getenv("INFLUX_USER"),
		Password: os.Getenv("INFLUX_PWD"),
	})
	if err != nil {
		fmt.Println("Error creating InfluxDB Client: ", err.Error())
	}
}

// Write a point using the UDP client
func ExampleClient_uDP() {
	// Make client
	config := client.UDPConfig{Addr: "localhost:8089"}
	c, err := client.NewUDPClient(config)
	if err != nil {
		fmt.Println("Error: ", err.Error())
	}
	defer c.Close()

	// Create a new point batch
	bp, _ := client.NewBatchPoints(client.BatchPointsConfig{
		Precision: "s",
	})

	// Create a point and add to batch
	tags := map[string]string{"cpu": "cpu-total"}
	fields := map[string]interface{}{
		"idle":   10.1,
		"system": 53.3,
		"user":   46.6,
	}
	pt, err := client.NewPoint("cpu_usage", tags, fields, time.Now())
	if err != nil {
		fmt.Println("Error: ", err.Error())
	}
	bp.AddPoint(pt)

	// Write the batch
	c.Write(bp)
}

// Ping the cluster using the HTTP client
func ExampleClient_Ping() {
	// Make client
	c, err := client.NewHTTPClient(client.HTTPConfig{
		Addr: "http://localhost:8086",
	})
	if err != nil {
		fmt.Println("Error creating InfluxDB Client: ", err.Error())
	}
	defer c.Close()

	_, _, err = c.Ping(0)
	if err != nil {
		fmt.Println("Error pinging InfluxDB Cluster: ", err.Error())
	}
}

// Write a point using the HTTP client
func ExampleClient_write() {
	// Make client
	c, err := client.NewHTTPClient(client.HTTPConfig{
		Addr: "http://localhost:8086",
	})
	if err != nil {
		fmt.Println("Error creating InfluxDB Client: ", err.Error())
	}
	defer c.Close()

	// Create a new point batch
	bp, _ := client.NewBatchPoints(client.BatchPointsConfig{
		Database:  "BumbleBeeTuna",
		Precision: "s",
	})

	// Create a point and add to batch
	tags := map[string]string{"cpu": "cpu-total"}
	fields := map[string]interface{}{
		"idle":   10.1,
		"system": 53.3,
		"user":   46.6,
	}
	pt, err := client.NewPoint("cpu_usage", tags, fields, time.Now())
	if err != nil {
		fmt.Println("Error: ", err.Error())
	}
	bp.AddPoint(pt)

	// Write the batch
	c.Write(bp)
}

// Create a batch and add a point
func ExampleBatchPoints() {
	// Create a new point batch
	bp, _ := client.NewBatchPoints(client.BatchPointsConfig{
		Database:  "BumbleBeeTuna",
		Precision: "s",
	})

	// Create a point and add to batch
	tags := map[string]string{"cpu": "cpu-total"}
	fields := map[string]interface{}{
		"idle":   10.1,
		"system": 53.3,
		"user":   46.6,
	}
	pt, err := client.NewPoint("cpu_usage", tags, fields, time.Now())
	if err != nil {
		fmt.Println("Error: ", err.Error())
	}
	bp.AddPoint(pt)
}

// Using the BatchPoints setter functions
func ExampleBatchPoints_setters() {
	// Create a new point batch
	bp, _ := client.NewBatchPoints(client.BatchPointsConfig{})
	bp.SetDatabase("BumbleBeeTuna")
	bp.SetPrecision("ms")

	// Create a point and add to batch
	tags := map[string]string{"cpu": "cpu-total"}
	fields := map[string]interface{}{
		"idle":   10.1,
		"system": 53.3,
		"user":   46.6,
	}
	pt, err := client.NewPoint("cpu_usage", tags, fields, time.Now())
	if err != nil {
		fmt.Println("Error: ", err.Error())
	}
	bp.AddPoint(pt)
}

// Create a new point with a timestamp
func ExamplePoint() {
	tags := map[string]string{"cpu": "cpu-total"}
	fields := map[string]interface{}{
		"idle":   10.1,
		"system": 53.3,
		"user":   46.6,
	}
	pt, err := client.NewPoint("cpu_usage", tags, fields, time.Now())
	if err == nil {
		fmt.Println("We created a point: ", pt.String())
	}
}

// Create a new point without a timestamp
func ExamplePoint_withoutTime() {
	tags := map[string]string{"cpu": "cpu-total"}
	fields := map[string]interface{}{
		"idle":   10.1,
		"system": 53.3,
		"user":   46.6,
	}
	pt, err := client.NewPoint("cpu_usage", tags, fields)
	if err == nil {
		fmt.Println("We created a point w/o time: ", pt.String())
	}
}

// Write 1000 points
func ExampleClient_write1000() {
	sampleSize := 1000

	// Make client
	c, err := client.NewHTTPClient(client.HTTPConfig{
		Addr: "http://localhost:8086",
	})
	if err != nil {
		fmt.Println("Error creating InfluxDB Client: ", err.Error())
	}
	defer c.Close()

	rand.Seed(42)

	bp, _ := client.NewBatchPoints(client.BatchPointsConfig{
		Database:  "systemstats",
		Precision: "us",
	})

	for i := 0; i < sampleSize; i++ {
		regions := []string{"us-west1", "us-west2", "us-west3", "us-east1"}
		tags := map[string]string{
			"cpu":    "cpu-total",
			"host":   fmt.Sprintf("host%d", rand.Intn(1000)),
			"region": regions[rand.Intn(len(regions))],
		}

		idle := rand.Float64() * 100.0
		fields := map[string]interface{}{
			"idle": idle,
			"busy": 100.0 - idle,
		}

		pt, err := client.NewPoint(
			"cpu_usage",
			tags,
			fields,
			time.Now(),
		)
		if err != nil {
			println("Error:", err.Error())
			continue
		}
		bp.AddPoint(pt)
	}

	err = c.Write(bp)
	if err != nil {
		fmt.Println("Error: ", err.Error())
	}
}

// Make a Query
func ExampleClient_query() {
	// Make client
	c, err := client.NewHTTPClient(client.HTTPConfig{
		Addr: "http://localhost:8086",
	})
	if err != nil {
		fmt.Println("Error creating InfluxDB Client: ", err.Error())
	}
	defer c.Close()

	q := client.NewQuery("SELECT count(value) FROM shapes", "square_holes", "ns")
	if response, err := c.Query(q); err == nil && response.Error() == nil {
		fmt.Println(response.Results)
	}
}

// Create a Database with a query
func ExampleClient_createDatabase() {
	// Make client
	c, err := client.NewHTTPClient(client.HTTPConfig{
		Addr: "http://localhost:8086",
	})
	if err != nil {
		fmt.Println("Error creating InfluxDB Client: ", err.Error())
	}
	defer c.Close()

	q := client.NewQuery("CREATE DATABASE telegraf", "", "")
	if response, err := c.Query(q); err == nil && response.Error() == nil {
		fmt.Println(response.Results)
	}
}
