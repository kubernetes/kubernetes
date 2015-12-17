package tsdb

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
)

// ShowMeasurementsExecutor implements the Executor interface for a SHOW MEASUREMENTS statement.
type ShowMeasurementsExecutor struct {
	stmt      *influxql.ShowMeasurementsStatement
	mappers   []Mapper
	chunkSize int
}

// NewShowMeasurementsExecutor returns a new ShowMeasurementsExecutor.
func NewShowMeasurementsExecutor(stmt *influxql.ShowMeasurementsStatement, mappers []Mapper, chunkSize int) *ShowMeasurementsExecutor {
	return &ShowMeasurementsExecutor{
		stmt:      stmt,
		mappers:   mappers,
		chunkSize: chunkSize,
	}
}

// Execute begins execution of the query and returns a channel to receive rows.
func (e *ShowMeasurementsExecutor) Execute(closing <-chan struct{}) <-chan *models.Row {
	// Create output channel and stream data in a separate goroutine.
	out := make(chan *models.Row, 0)

	go func() {
		// It's important that all resources are released when execution completes.
		defer e.close()

		// Open the mappers.
		for _, m := range e.mappers {
			if err := m.Open(); err != nil {
				out <- &models.Row{Err: err}
				return
			}
		}

		// Create a set to hold measurement names from mappers.
		set := map[string]struct{}{}
		// Iterate through mappers collecting measurement names.
		for _, m := range e.mappers {
			// Get the data from the mapper.
			c, err := m.NextChunk()
			if err != nil {
				out <- &models.Row{Err: err}
				return
			} else if c == nil {
				// Mapper had no data.
				continue
			}

			// Convert the mapper chunk to MapperOutput type.
			mop, ok := c.(*MapperOutput)
			if !ok {
				out <- &models.Row{Err: fmt.Errorf("show measurements mapper returned invalid type: %T", c)}
				return
			}

			// Add the measurement names to the set.
			for _, mv := range mop.Values {
				mm, ok := mv.Value.(string)
				if !ok {
					out <- &models.Row{Err: fmt.Errorf("show measurements mapper returned invalid type: %T", mop)}
					return
				}
				set[mm] = struct{}{}
			}
		}

		// Convert the set into an array of measurement names.
		measurements := make([]string, 0, len(set))
		for mm := range set {
			measurements = append(measurements, mm)
		}
		// Sort the names.
		sort.Strings(measurements)

		// Calculate OFFSET and LIMIT
		off := e.stmt.Offset
		lim := len(measurements)
		stmtLim := e.stmt.Limit

		if stmtLim > 0 && off+stmtLim < lim {
			lim = off + stmtLim
		} else if off > lim {
			off, lim = 0, 0
		}

		// Put the results in a row and send it.
		row := &models.Row{
			Name:    "measurements",
			Columns: []string{"name"},
			Values:  make([][]interface{}, 0, len(measurements)),
		}

		for _, m := range measurements[off:lim] {
			v := []interface{}{m}
			row.Values = append(row.Values, v)
		}

		if len(row.Values) > 0 {
			select {
			case out <- row:
			case <-closing:
				out <- &models.Row{Err: fmt.Errorf("execute was closed by caller")}
				break
			case <-time.After(30 * time.Second):
				// This should never happen, so if it does, it is a problem
				out <- &models.Row{Err: fmt.Errorf("execute was closed by read timeout")}
				break
			}
		}

		close(out)
	}()
	return out
}

// Close closes the executor such that all resources are released. Once closed,
// an executor may not be re-used.
func (e *ShowMeasurementsExecutor) close() {
	if e != nil {
		for _, m := range e.mappers {
			m.Close()
		}
	}
}

// ShowMeasurementsMapper is a mapper for collecting measurement names from a shard.
type ShowMeasurementsMapper struct {
	remote Mapper
	shard  *Shard
	stmt   *influxql.ShowMeasurementsStatement
	state  interface{}

	ChunkSize int
}

// NewShowMeasurementsMapper returns a mapper for the given shard, which will return data for the meta statement.
func NewShowMeasurementsMapper(shard *Shard, stmt *influxql.ShowMeasurementsStatement) *ShowMeasurementsMapper {
	return &ShowMeasurementsMapper{
		shard: shard,
		stmt:  stmt,
	}
}

// Open opens the mapper for use.
func (m *ShowMeasurementsMapper) Open() error {
	if m.remote != nil {
		return m.remote.Open()
	}

	var measurements Measurements

	if m.shard != nil {
		// If a WHERE clause was specified, filter the measurements.
		if m.stmt.Condition != nil {
			var err error
			measurements, err = m.shard.index.measurementsByExpr(m.stmt.Condition)
			if err != nil {
				return err
			}
		} else {
			// Otherwise, get all measurements from the database.
			measurements = m.shard.index.Measurements()
		}
		sort.Sort(measurements)
	}

	// Create a channel to send measurement names on.
	ch := make(chan string)
	// Start a goroutine to send the names over the channel as needed.
	go func() {
		for _, mm := range measurements {
			// Filter measurements by WITH clause, if one was given.
			if m.stmt.Source != nil {
				s, ok := m.stmt.Source.(*influxql.Measurement)
				if !ok ||
					s.Regex != nil && !s.Regex.Val.MatchString(mm.Name) ||
					s.Name != "" && s.Name != mm.Name {
					continue
				}
			}

			ch <- mm.Name
		}
		close(ch)
	}()

	// Store the channel as the state of the mapper.
	m.state = ch

	return nil
}

// SetRemote sets the remote mapper to use.
func (m *ShowMeasurementsMapper) SetRemote(remote Mapper) { m.remote = remote }

// TagSets is only implemented on this mapper to satisfy the Mapper interface.
func (m *ShowMeasurementsMapper) TagSets() []string { return nil }

// Fields returns a list of field names for this mapper.
func (m *ShowMeasurementsMapper) Fields() []string { return []string{"name"} }

// NextChunk returns the next chunk of measurement names.
func (m *ShowMeasurementsMapper) NextChunk() (interface{}, error) {
	if m.remote != nil {
		b, err := m.remote.NextChunk()
		if err != nil {
			return nil, err
		} else if b == nil {
			return nil, nil
		}

		mop := &MapperOutput{
			Name:   "measurements",
			Fields: []string{"name"},
			Values: make([]*MapperValue, 0),
		}

		if err := json.Unmarshal(b.([]byte), &mop); err != nil {
			return nil, err
		}

		return mop, nil
	}
	return m.nextChunk()
}

// nextChunk implements next chunk logic for a local shard.
func (m *ShowMeasurementsMapper) nextChunk() (interface{}, error) {
	var output *MapperOutput

	names := make([]string, 0, m.ChunkSize)

	// Get the channel of measurement names from the state.
	measurementNames := m.state.(chan string)

	// Get the next chunk of names.
	for n := range measurementNames {
		names = append(names, n)
		if len(names) == m.ChunkSize {
			break
		}
	}

	output = &MapperOutput{
		Name:   "measurements",
		Fields: []string{"name"},
		Values: make([]*MapperValue, 0, len(names)),
	}

	for _, v := range names {
		output.Values = append(output.Values, &MapperValue{
			Value: v,
		})
	}

	return output, nil
}

// Close closes the mapper.
func (m *ShowMeasurementsMapper) Close() {
	if m.remote != nil {
		m.remote.Close()
	}
}
