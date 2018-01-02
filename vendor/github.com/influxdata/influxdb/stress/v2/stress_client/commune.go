package stressClient

import (
	"log"
	"time"

	"github.com/influxdata/influxdb/models"
)

// Communes are a method for passing points between InsertStatements and QueryStatements.

type commune struct {
	ch          chan string
	storedPoint models.Point
}

// NewCommune creates a new commune with a buffered chan of length n
func newCommune(n int) *commune {
	return &commune{ch: make(chan string, n)}
}

func (c *commune) point(precision string) models.Point {

	pt := []byte(<-c.ch)

	p, err := models.ParsePointsWithPrecision(pt, time.Now().UTC(), precision)

	if err != nil {
		log.Fatalf("Error parsing point for commune\n  point: %v\n  error: %v\n", pt, err)
	}

	if len(p) == 0 {
		return c.storedPoint
	}

	c.storedPoint = p[0]
	return p[0]
}

// SetCommune creates a new commune on the StressTest
func (st *StressTest) SetCommune(name string) chan<- string {
	com := newCommune(10)
	st.communes[name] = com

	return com.ch
}

// GetPoint is called by a QueryStatement and retrieves a point sent by the associated InsertStatement
func (st *StressTest) GetPoint(name, precision string) models.Point {
	p := st.communes[name].point(precision)

	// Function needs to return a point. Panic if it doesn't
	if p == nil {
		log.Fatal("Commune not returning point")
	}

	return p
}
