package monitor

import (
	"fmt"
	"sort"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
)

// StatementExecutor translates InfluxQL queries to Monitor methods.
type StatementExecutor struct {
	Monitor interface {
		Statistics(map[string]string) ([]*Statistic, error)
		Diagnostics() (map[string]*Diagnostic, error)
	}
}

// ExecuteStatement executes monitor-related query statements.
func (s *StatementExecutor) ExecuteStatement(stmt influxql.Statement) *influxql.Result {
	switch stmt := stmt.(type) {
	case *influxql.ShowStatsStatement:
		return s.executeShowStatistics(stmt.Module)
	case *influxql.ShowDiagnosticsStatement:
		return s.executeShowDiagnostics(stmt.Module)
	default:
		panic(fmt.Sprintf("unsupported statement type: %T", stmt))
	}
}

func (s *StatementExecutor) executeShowStatistics(module string) *influxql.Result {
	stats, err := s.Monitor.Statistics(nil)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	var rows []*models.Row
	for _, stat := range stats {
		if module != "" && stat.Name != module {
			continue
		}
		row := &models.Row{Name: stat.Name, Tags: stat.Tags}

		values := make([]interface{}, 0, len(stat.Values))
		for _, k := range stat.valueNames() {
			row.Columns = append(row.Columns, k)
			values = append(values, stat.Values[k])
		}
		row.Values = [][]interface{}{values}
		rows = append(rows, row)
	}
	return &influxql.Result{Series: rows}
}

func (s *StatementExecutor) executeShowDiagnostics(module string) *influxql.Result {
	diags, err := s.Monitor.Diagnostics()
	if err != nil {
		return &influxql.Result{Err: err}
	}
	rows := make([]*models.Row, 0, len(diags))

	// Get a sorted list of diagnostics keys.
	sortedKeys := make([]string, 0, len(diags))
	for k := range diags {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Strings(sortedKeys)

	for _, k := range sortedKeys {
		if module != "" && k != module {
			continue
		}

		row := &models.Row{Name: k}

		row.Columns = diags[k].Columns
		row.Values = diags[k].Rows
		rows = append(rows, row)
	}
	return &influxql.Result{Series: rows}
}
