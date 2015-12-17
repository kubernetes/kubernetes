package tsdb

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
)

// ShowTagKeysExecutor implements the Executor interface for a SHOW MEASUREMENTS statement.
type ShowTagKeysExecutor struct {
	stmt      *influxql.ShowTagKeysStatement
	mappers   []Mapper
	chunkSize int
}

// NewShowTagKeysExecutor returns a new ShowTagKeysExecutor.
func NewShowTagKeysExecutor(stmt *influxql.ShowTagKeysStatement, mappers []Mapper, chunkSize int) *ShowTagKeysExecutor {
	return &ShowTagKeysExecutor{
		stmt:      stmt,
		mappers:   mappers,
		chunkSize: chunkSize,
	}
}

// Execute begins execution of the query and returns a channel to receive rows.
func (e *ShowTagKeysExecutor) Execute(closing <-chan struct{}) <-chan *models.Row {
	// It's important that all resources are released when execution completes.
	defer e.close()

	// Create output channel and stream data in a separate goroutine.
	out := make(chan *models.Row, 0)

	go func() {
		defer close(out)
		// Open the mappers.
		for _, m := range e.mappers {
			if err := m.Open(); err != nil {
				out <- &models.Row{Err: err}
				return
			}
		}

		// Create a map of measurement to tags keys.
		set := map[string]map[string]struct{}{}
		// Iterate through mappers collecting measurement names.
		for _, m := range e.mappers {
			// Read all data from the mapper.
			for {
				c, err := m.NextChunk()
				if err != nil {
					out <- &models.Row{Err: err}
					return
				} else if c == nil {
					// Mapper has been drained.
					break
				}

				// Convert the mapper chunk to an array of measurements with tag keys.
				mtks, ok := c.(MeasurementsTagKeys)
				if !ok {
					out <- &models.Row{Err: fmt.Errorf("show tag keys mapper returned invalid type: %T", c)}
					return
				}

				// Merge mapper chunk with previous mapper outputs.
				for _, mm := range mtks {
					for _, key := range mm.TagKeys {
						if set[mm.Measurement] == nil {
							set[mm.Measurement] = map[string]struct{}{}
						}
						set[mm.Measurement][key] = struct{}{}
					}
				}
			}
		}

		// All mappers are drained.

		// Convert the set into an array of measurements and their tag keys.
		mstks := make(MeasurementsTagKeys, 0)
		for mm, tks := range set {
			mtks := &MeasurementTagKeys{Measurement: mm}
			for tk := range tks {
				mtks.TagKeys = append(mtks.TagKeys, tk)
			}
			sort.Strings(mtks.TagKeys)
			mstks = append(mstks, mtks)
		}
		// Sort by measurement name.
		sort.Sort(mstks)

		slim, soff := limitAndOffset(e.stmt.SLimit, e.stmt.SOffset, len(mstks))

		// Send results.
		for _, mtks := range mstks[soff:slim] {
			lim, off := limitAndOffset(e.stmt.Limit, e.stmt.Offset, len(mtks.TagKeys))

			row := &models.Row{
				Name:    mtks.Measurement,
				Columns: []string{"tagKey"},
				Values:  make([][]interface{}, 0, lim-off),
			}

			for _, tk := range mtks.TagKeys[off:lim] {
				v := []interface{}{tk}
				row.Values = append(row.Values, v)
			}

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
	}()
	return out
}

// limitAndOffset calculates the limit and offset indexes for n things.
func limitAndOffset(lim, off, n int) (int, int) {
	if off >= n {
		return 0, 0
	}

	o := off
	l := n

	if lim > 0 && o+lim < l {
		l = o + lim
	}

	if o > l {
		return 0, 0
	}

	return l, o
}

// Close closes the executor such that all resources are released. Once closed,
// an executor may not be re-used.
func (e *ShowTagKeysExecutor) close() {
	if e != nil {
		for _, m := range e.mappers {
			m.Close()
		}
	}
}

// ShowTagKeysMapper is a mapper for collecting measurement names from a shard.
type ShowTagKeysMapper struct {
	remote    Mapper
	shard     *Shard
	stmt      *influxql.ShowTagKeysStatement
	chunkSize int
	state     interface{}
}

// NewShowTagKeysMapper returns a mapper for the given shard, which will return data for the meta statement.
func NewShowTagKeysMapper(shard *Shard, stmt *influxql.ShowTagKeysStatement, chunkSize int) *ShowTagKeysMapper {
	return &ShowTagKeysMapper{
		shard:     shard,
		stmt:      stmt,
		chunkSize: chunkSize,
	}
}

// MeasurementTagKeys represents measurement tag keys.
type MeasurementTagKeys struct {
	Measurement string   `json:"measurement"`
	TagKeys     []string `json:"tagkeys"`
}

// MeasurementsTagKeys represents tag keys for multiple measurements.
type MeasurementsTagKeys []*MeasurementTagKeys

func (a MeasurementsTagKeys) Len() int           { return len(a) }
func (a MeasurementsTagKeys) Less(i, j int) bool { return a[i].Measurement < a[j].Measurement }
func (a MeasurementsTagKeys) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// Size returns the total string length of measurement names & tag keys.
func (a MeasurementsTagKeys) Size() int {
	n := 0
	for _, m := range a {
		n += len(m.Measurement)
		for _, k := range m.TagKeys {
			n += len(k)
		}
	}
	return n
}

// Open opens the mapper for use.
func (m *ShowTagKeysMapper) Open() error {
	if m.remote != nil {
		return m.remote.Open()
	}

	// This can happen when a shard has been assigned to this node but we have not
	// written to it so it may not exist yet.
	if m.shard == nil {
		return nil
	}

	sources := influxql.Sources{}

	// Expand regex expressions in the FROM clause.
	if m.stmt.Sources != nil {
		var err error
		sources, err = m.shard.index.ExpandSources(m.stmt.Sources)
		if err != nil {
			return err
		}
	}

	// Get measurements from sources in the statement if provided or database if not.
	measurements, err := measurementsFromSourcesOrDB(m.shard.index, sources...)
	if err != nil {
		return err
	}

	// If a WHERE clause was specified, filter the measurements.
	if m.stmt.Condition != nil {
		var err error
		whereMs, err := m.shard.index.measurementsByExpr(m.stmt.Condition)
		if err != nil {
			return err
		}

		sort.Sort(whereMs)

		measurements = measurements.intersect(whereMs)
	}

	// Create a channel to send measurement names on.
	ch := make(chan *MeasurementTagKeys)
	// Start a goroutine to send the names over the channel as needed.
	go func() {
		for _, mm := range measurements {
			ch <- &MeasurementTagKeys{
				Measurement: mm.Name,
				TagKeys:     mm.TagKeys(),
			}
		}
		close(ch)
	}()

	// Store the channel as the state of the mapper.
	m.state = ch

	return nil
}

// SetRemote sets the remote mapper to use.
func (m *ShowTagKeysMapper) SetRemote(remote Mapper) error {
	m.remote = remote
	return nil
}

// TagSets is only implemented on this mapper to satisfy the Mapper interface.
func (m *ShowTagKeysMapper) TagSets() []string { return nil }

// Fields returns a list of field names for this mapper.
func (m *ShowTagKeysMapper) Fields() []string { return []string{"tagKey"} }

// NextChunk returns the next chunk of measurements and tag keys.
func (m *ShowTagKeysMapper) NextChunk() (interface{}, error) {
	if m.remote != nil {
		b, err := m.remote.NextChunk()
		if err != nil {
			return nil, err
		} else if b == nil {
			return nil, nil
		}

		mtks := []*MeasurementTagKeys{}
		if err := json.Unmarshal(b.([]byte), &mtks); err != nil {
			return nil, err
		} else if len(mtks) == 0 {
			// Mapper on other node sent 0 values so it's done.
			return nil, nil
		}
		return mtks, nil
	}
	return m.nextChunk()
}

// nextChunk implements next chunk logic for a local shard.
func (m *ShowTagKeysMapper) nextChunk() (interface{}, error) {
	// Get the channel of measurement tag keys from the state.
	ch, ok := m.state.(chan *MeasurementTagKeys)
	if !ok {
		return nil, nil
	}
	// Allocate array to hold measurement names.
	mtks := make(MeasurementsTagKeys, 0)
	// Get the next chunk of tag keys.
	for n := range ch {
		mtks = append(mtks, n)
		if mtks.Size() >= m.chunkSize {
			break
		}
	}
	// See if we've read all the names.
	if len(mtks) == 0 {
		return nil, nil
	}

	return mtks, nil
}

// Close closes the mapper.
func (m *ShowTagKeysMapper) Close() {
	if m.remote != nil {
		m.remote.Close()
	}
}
