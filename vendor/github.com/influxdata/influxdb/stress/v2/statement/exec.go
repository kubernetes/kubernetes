package statement

import (
	"time"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

// ExecStatement run outside scripts. This functionality is not built out
// TODO: Wire up!
type ExecStatement struct {
	StatementID string
	Script      string

	runtime time.Duration
}

// SetID statisfies the Statement Interface
func (i *ExecStatement) SetID(s string) {
	i.StatementID = s
}

// Run statisfies the Statement Interface
func (i *ExecStatement) Run(s *stressClient.StressTest) {
	runtime := time.Now()
	i.runtime = time.Since(runtime)
}

// Report statisfies the Statement Interface
func (i *ExecStatement) Report(s *stressClient.StressTest) string {
	return ""
}
