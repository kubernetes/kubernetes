package statement

import (
	"log"
	"strconv"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

// Statement is the common interface to shape the testing environment and prepare database requests
// The parser turns the 'statements' in the config file into Statements
type Statement interface {
	Run(s *stressClient.StressTest)
	Report(s *stressClient.StressTest) string
	SetID(s string)
}

func parseInt(s string) int {
	i, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		log.Fatalf("Error parsing integer:\n  String: %v\n  Error: %v\n", s, err)
	}
	return int(i)
}

func parseFloat(s string) int {
	i, err := strconv.ParseFloat(s, 64)
	if err != nil {
		log.Fatalf("Error parsing integer:\n  String: %v\n  Error: %v\n", s, err)
	}
	return int(i)
}
