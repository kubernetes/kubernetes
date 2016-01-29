package tsdb

// Monitor represents a TSDB monitoring service.
type Monitor struct {
	Store interface{}
}

func (m *Monitor) Open() error  { return nil }
func (m *Monitor) Close() error { return nil }

// StartSelfMonitoring starts a goroutine which monitors the InfluxDB server
// itself and stores the results in the specified database at a given interval.
/*
func (s *Server) StartSelfMonitoring(database, retention string, interval time.Duration) error {
		if interval == 0 {
			return fmt.Errorf("statistics check interval must be non-zero")
		}

		go func() {
			tick := time.NewTicker(interval)
			for {
				<-tick.C

				// Create the batch and tags
				tags := map[string]string{"serverID": strconv.FormatUint(s.ID(), 10)}
				if h, err := os.Hostname(); err == nil {
					tags["host"] = h
				}
				batch := pointsFromStats(s.stats, tags)

				// Shard-level stats.
				tags["shardID"] = strconv.FormatUint(s.id, 10)
				s.mu.RLock()
				for _, sh := range s.shards {
					if !sh.HasDataNodeID(s.id) {
						// No stats for non-local shards.
						continue
					}
					batch = append(batch, pointsFromStats(sh.stats, tags)...)
				}
				s.mu.RUnlock()

				// Server diagnostics.
				for _, row := range s.DiagnosticsAsRows() {
					points, err := s.convertRowToPoints(row.Name, row)
					if err != nil {
						s.Logger.Printf("failed to write diagnostic row for %s: %s", row.Name, err.Error())
						continue
					}
					for _, p := range points {
						p.AddTag("serverID", strconv.FormatUint(s.ID(), 10))
					}
					batch = append(batch, points...)
				}

				s.WriteSeries(database, retention, batch)
			}
		}()
	return nil
}

// Function for local use turns stats into a slice of points
func pointsFromStats(st *Stats, tags map[string]string) []tsdb.Point {
	var points []tsdb.Point
	now := time.Now()
	st.Walk(func(k string, v int64) {
		point := tsdb.NewPoint(
			st.name+"_"+k,
			make(map[string]string),
			map[string]interface{}{"value": int(v)},
			now,
		)
		// Specifically create a new map.
		for k, v := range tags {
			tags[k] = v
			point.AddTag(k, v)
		}
		points = append(points, point)
	})

	return points
}
*/
