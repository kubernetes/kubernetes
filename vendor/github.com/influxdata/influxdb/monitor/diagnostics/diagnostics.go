package diagnostics // import "github.com/influxdata/influxdb/monitor/diagnostics"

// Client is the interface modules implement if they register diagnostics with monitor.
type Client interface {
	Diagnostics() (*Diagnostics, error)
}

// The ClientFunc type is an adapter to allow the use of
// ordinary functions as Diagnostics clients.
type ClientFunc func() (*Diagnostics, error)

// Diagnostics calls f().
func (f ClientFunc) Diagnostics() (*Diagnostics, error) {
	return f()
}

// Diagnostics represents a table of diagnostic information. The first value
// is the name of the columns, the second is a slice of interface slices containing
// the values for each column, by row. This information is never written to an InfluxDB
// system and is display-only. An example showing, say, connections follows:
//
//     source_ip    source_port       dest_ip     dest_port
//     182.1.0.2    2890              127.0.0.1   38901
//     174.33.1.2   2924              127.0.0.1   38902
type Diagnostics struct {
	Columns []string
	Rows    [][]interface{}
}

// NewDiagnostic initialises a new Diagnostics with the specified columns.
func NewDiagnostics(columns []string) *Diagnostics {
	return &Diagnostics{
		Columns: columns,
		Rows:    make([][]interface{}, 0),
	}
}

// AddRow appends the provided row to the Diagnostics' rows.
func (d *Diagnostics) AddRow(r []interface{}) {
	d.Rows = append(d.Rows, r)
}
