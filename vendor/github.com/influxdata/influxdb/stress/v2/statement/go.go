package statement

import (
	"fmt"
	"time"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

// GoStatement is a Statement Implementation to allow other statements to be run concurrently
type GoStatement struct {
	Statement

	StatementID string
}

// SetID statisfies the Statement Interface
func (i *GoStatement) SetID(s string) {
	i.StatementID = s
}

// Run statisfies the Statement Interface
func (i *GoStatement) Run(s *stressClient.StressTest) {
	// TODO: remove
	switch i.Statement.(type) {
	case *QueryStatement:
		time.Sleep(1 * time.Second)
	}

	s.Add(1)
	go func() {
		i.Statement.Run(s)
		s.Done()
	}()
}

// Report statisfies the Statement Interface
func (i *GoStatement) Report(s *stressClient.StressTest) string {
	return fmt.Sprintf("Go %v", i.Statement.Report(s))
}
