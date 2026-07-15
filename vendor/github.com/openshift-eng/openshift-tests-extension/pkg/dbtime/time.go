package dbtime

import "time"

// DBTime is a type suitable for direct importing into databases like BigQuery,
// formatted like 2006-01-02 15:04:05.000000 UTC.
type DBTime time.Time

func Ptr(t time.Time) *DBTime {
	return (*DBTime)(&t)
}

func (dbt *DBTime) MarshalJSON() ([]byte, error) {
	formattedTime := time.Time(*dbt).Format(`"2006-01-02 15:04:05.000000 UTC"`)
	return []byte(formattedTime), nil
}

func (dbt *DBTime) UnmarshalJSON(b []byte) error {
	timeStr := string(b[1 : len(b)-1])
	parsedTime, err := time.Parse("2006-01-02 15:04:05.000000 UTC", timeStr)
	if err != nil {
		return err
	}
	*dbt = (DBTime)(parsedTime)
	return nil
}
