package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/influxdata/influxdb/cmd/influx_tsm/stats"
	"github.com/influxdata/influxdb/cmd/influx_tsm/tsdb"
)

// tracker will orchestrate and track the conversions of non-TSM shards to TSM
type tracker struct {
	Stats stats.Stats

	shards tsdb.ShardInfos
	opts   options

	pg ParallelGroup
	wg sync.WaitGroup
}

// newTracker will setup and return a clean tracker instance
func newTracker(shards tsdb.ShardInfos, opts options) *tracker {
	t := &tracker{
		shards: shards,
		opts:   opts,
		pg:     NewParallelGroup(runtime.GOMAXPROCS(0)),
	}

	return t
}

func (t *tracker) Run() error {
	conversionStart := time.Now()

	// Backup each directory.
	if !opts.SkipBackup {
		databases := t.shards.Databases()
		fmt.Printf("Backing up %d databases...\n", len(databases))
		t.wg.Add(len(databases))
		for i := range databases {
			db := databases[i]
			go t.pg.Do(func() {
				defer t.wg.Done()

				start := time.Now()
				log.Printf("Backup of database '%v' started", db)
				err := backupDatabase(db)
				if err != nil {
					log.Fatalf("Backup of database %v failed: %v\n", db, err)
				}
				log.Printf("Database %v backed up (%v)\n", db, time.Now().Sub(start))
			})
		}
		t.wg.Wait()
	} else {
		fmt.Println("Database backup disabled.")
	}

	t.wg.Add(len(t.shards))
	for i := range t.shards {
		si := t.shards[i]
		go t.pg.Do(func() {
			defer func() {
				atomic.AddUint64(&t.Stats.CompletedShards, 1)
				t.wg.Done()
			}()

			start := time.Now()
			log.Printf("Starting conversion of shard: %v", si.FullPath(opts.DataPath))
			if err := convertShard(si, t); err != nil {
				log.Fatalf("Failed to convert %v: %v\n", si.FullPath(opts.DataPath), err)
			}
			log.Printf("Conversion of %v successful (%v)\n", si.FullPath(opts.DataPath), time.Since(start))
		})
	}

	done := make(chan struct{})
	go func() {
		t.wg.Wait()
		close(done)
	}()

WAIT_LOOP:
	for {
		select {
		case <-done:
			break WAIT_LOOP
		case <-time.After(opts.UpdateInterval):
			t.StatusUpdate()
		}
	}

	t.Stats.TotalTime = time.Since(conversionStart)

	return nil
}

func (t *tracker) StatusUpdate() {
	shardCount := atomic.LoadUint64(&t.Stats.CompletedShards)
	pointCount := atomic.LoadUint64(&t.Stats.PointsRead)
	pointWritten := atomic.LoadUint64(&t.Stats.PointsWritten)

	log.Printf("Still Working: Completed Shards: %d/%d Points read/written: %d/%d", shardCount, len(t.shards), pointCount, pointWritten)
}

func (t *tracker) PrintStats() {
	preSize := t.shards.Size()
	postSize := int64(t.Stats.TsmBytesWritten)

	fmt.Printf("\nSummary statistics\n========================================\n")
	fmt.Printf("Databases converted:                 %d\n", len(t.shards.Databases()))
	fmt.Printf("Shards converted:                    %d\n", len(t.shards))
	fmt.Printf("TSM files created:                   %d\n", t.Stats.TsmFilesCreated)
	fmt.Printf("Points read:                         %d\n", t.Stats.PointsRead)
	fmt.Printf("Points written:                      %d\n", t.Stats.PointsWritten)
	fmt.Printf("NaN filtered:                        %d\n", t.Stats.NanFiltered)
	fmt.Printf("Inf filtered:                        %d\n", t.Stats.InfFiltered)
	fmt.Printf("Points without fields filtered:      %d\n", t.Stats.FieldsFiltered)
	fmt.Printf("Disk usage pre-conversion (bytes):   %d\n", preSize)
	fmt.Printf("Disk usage post-conversion (bytes):  %d\n", postSize)
	fmt.Printf("Reduction factor:                    %d%%\n", 100*(preSize-postSize)/preSize)
	fmt.Printf("Bytes per TSM point:                 %.2f\n", float64(postSize)/float64(t.Stats.PointsWritten))
	fmt.Printf("Total conversion time:               %v\n", t.Stats.TotalTime)
	fmt.Println()
}
