// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"container/list"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"code.google.com/p/google-api-go-client/bigquery/v2"
	"code.google.com/p/google-api-go-client/storage/v1beta2"
)

const (
	GB                         = 1 << 30
	MaxBackoff                 = 30000
	BaseBackoff                = 250
	BackoffGrowthFactor        = 1.8
	BackoffGrowthDamper        = 0.25
	JobStatusDone              = "DONE"
	DatasetAlreadyExists       = "Already Exists: Dataset"
	TableWriteEmptyDisposition = "WRITE_EMPTY"
)

func init() {
	scope := fmt.Sprintf("%s %s %s", bigquery.BigqueryScope,
		storage.DevstorageRead_onlyScope,
		"https://www.googleapis.com/auth/userinfo.profile")
	registerDemo("bigquery", scope, bqMain)
}

// This example demonstrates loading objects from Google Cloud Storage into
// BigQuery. Objects are specified by their bucket and a name prefix. Each
// object will be loaded into a new table identified by the object name minus
// any file extension. All tables are added to the specified dataset (one will
// be created if necessary). Currently, tables will not be overwritten and an
// attempt to load an object into a dataset that already contains its table
// will emit an error message indicating the table already exists.
// A schema file must be provided and it will be applied to every object/table.
// Example usage:
//   go-api-demo -clientid="my-clientid" -secret="my-secret" bq myProject
//								myDataBucket datafile2013070 DataFiles2013
//								./datafile_schema.json 100
//
// This will load all objects (e.g. all data files from July 2013) from
// gs://myDataBucket into a (possibly new) BigQuery dataset named DataFiles2013
// using the schema file provided and allowing up to 100 bad records. Assuming
// each object is named like datafileYYYYMMDD.csv.gz and all of July's files are
// stored in the bucket, 9 tables will be created named like datafile201307DD
// where DD ranges from 01 to 09, inclusive.
// When the program completes, it will emit a results line similar to:
//
// 9 files loaded in 3m58s (18m2.708s). Size: 7.18GB Rows: 7130725
//
// The total elapsed time from the start of first job to the end of the last job
// (effectively wall clock time) is shown. In parenthesis is the aggregate time
// taken to load all tables.
func bqMain(client *http.Client, argv []string) {
	if len(argv) != 6 {
		fmt.Fprintln(os.Stderr,
			"Usage: bq project_id bucket prefix dataset schema max_bad_records")
		return
	}

	var (
		project    = argv[0]
		bucket     = argv[1]
		objPrefix  = argv[2]
		datasetId  = argv[3]
		schemaFile = argv[4]
	)
	badRecords, err := strconv.ParseInt(argv[5], 10, 64)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}

	rand.Seed(time.Now().UnixNano())

	service, err := storage.New(client)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}

	// Get the list of objects in the bucket matching the specified prefix.
	list := service.Objects.List(bucket)
	list.Prefix(objPrefix)
	objects, err := list.Do()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}

	// Create the wrapper and insert the (new) dataset.
	dataset, err := newBQDataset(client, project, datasetId)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	if err = dataset.insert(true); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}

	objectSource := &tableSource{
		maxBadRecords: badRecords,
		disposition:   TableWriteEmptyDisposition,
	}

	// Load the schema from disk.
	f, err := ioutil.ReadFile(schemaFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	if err = json.Unmarshal(f, &objectSource.schema); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}

	// Assumes all objects have .csv, .csv.gz (or no) extension.
	tableIdFromObject := func(name string) string {
		return strings.TrimSuffix(strings.TrimSuffix(name, ".gz"), ".csv")
	}

	// A jobset is way to group a collection of jobs together for monitoring.
	// For this example, we just use the name of the bucket and object prefix.
	jobset := fmt.Sprintf("%s:%s", bucket, objPrefix)
	fmt.Fprintf(os.Stderr, "\nLoading %d objects.\n", len(objects.Items))

	// Load each object into a dataset of the same name (minus any extension).
	// A successful insert call will inject the job into our queue for monitoring.
	for _, o := range objects.Items {
		objectSource.id = tableIdFromObject(o.Name)
		objectSource.uri = fmt.Sprintf("gs://%s/%s", o.Bucket, o.Name)
		if err = dataset.load(jobset, objectSource); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
	}

	dataset.monitor(jobset)
}

// Wraps the BigQuery service and dataset and provides some helper functions.
type bqDataset struct {
	project string
	id      string
	bq      *bigquery.Service
	dataset *bigquery.Dataset
	jobsets map[string]*list.List
}

func newBQDataset(client *http.Client, dsProj string, dsId string) (*bqDataset,
	error) {

	service, err := bigquery.New(client)
	if err != nil {
		return nil, err
	}

	return &bqDataset{
		project: dsProj,
		id:      dsId,
		bq:      service,
		dataset: &bigquery.Dataset{
			DatasetReference: &bigquery.DatasetReference{
				DatasetId: dsId,
				ProjectId: dsProj,
			},
		},
		jobsets: make(map[string]*list.List),
	}, nil
}

func (ds *bqDataset) insert(existsOK bool) error {
	call := ds.bq.Datasets.Insert(ds.project, ds.dataset)
	_, err := call.Do()
	if err != nil && (!existsOK || !strings.Contains(err.Error(),
		DatasetAlreadyExists)) {
		return err
	}

	return nil
}

type tableSource struct {
	id            string
	uri           string
	schema        bigquery.TableSchema
	maxBadRecords int64
	disposition   string
}

func (ds *bqDataset) load(jobset string, source *tableSource) error {
	job := &bigquery.Job{
		Configuration: &bigquery.JobConfiguration{
			Load: &bigquery.JobConfigurationLoad{
				DestinationTable: &bigquery.TableReference{
					DatasetId: ds.dataset.DatasetReference.DatasetId,
					ProjectId: ds.project,
					TableId:   source.id,
				},
				MaxBadRecords:    source.maxBadRecords,
				Schema:           &source.schema,
				SourceUris:       []string{source.uri},
				WriteDisposition: source.disposition,
			},
		},
	}

	call := ds.bq.Jobs.Insert(ds.project, job)
	job, err := call.Do()
	if err != nil {
		return err
	}

	_, ok := ds.jobsets[jobset]
	if !ok {
		ds.jobsets[jobset] = list.New()
	}
	ds.jobsets[jobset].PushBack(job)

	return nil
}

func (ds *bqDataset) getJob(id string) (*bigquery.Job, error) {
	return ds.bq.Jobs.Get(ds.project, id).Do()
}

func (ds *bqDataset) monitor(jobset string) {
	jobq, ok := ds.jobsets[jobset]
	if !ok {
		return
	}

	var backoff float64 = BaseBackoff
	pause := func(grow bool) {
		if grow {
			backoff *= BackoffGrowthFactor
			backoff -= (backoff * rand.Float64() * BackoffGrowthDamper)
			backoff = math.Min(backoff, MaxBackoff)
			fmt.Fprintf(os.Stderr, "[%s] Checking remaining %d jobs...\n", jobset,
				1+jobq.Len())
		}
		time.Sleep(time.Duration(backoff) * time.Millisecond)
	}
	var stats jobStats

	// Track a 'head' pending job in queue for detecting cycling.
	head := ""
	// Loop until all jobs are done - with either success or error.
	for jobq.Len() > 0 {
		jel := jobq.Front()
		job := jel.Value.(*bigquery.Job)
		jobq.Remove(jel)
		jid := job.JobReference.JobId
		loop := false

		// Check and possibly pick a new head job id.
		if len(head) == 0 {
			head = jid
		} else {
			if jid == head {
				loop = true
			}
		}

		// Retrieve the job's current status.
		pause(loop)
		j, err := ds.getJob(jid)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			// In this case of a transient API error, we want keep the job.
			if j == nil {
				jobq.PushBack(job)
			} else {
				// Must reset head tracker if job is discarded.
				if loop {
					head = ""
					backoff = BaseBackoff
				}
			}
			continue
		}

		// Reassign with the updated job data (from Get).
		// We don't use j here as Get might return nil for this value.
		job = j

		if job.Status.State != JobStatusDone {
			jobq.PushBack(job)
			continue
		}

		if res := job.Status.ErrorResult; res != nil {
			fmt.Fprintln(os.Stderr, res.Message)
		} else {
			stat := job.Statistics
			lstat := stat.Load
			stats.files += 1
			stats.bytesIn += lstat.InputFileBytes
			stats.bytesOut += lstat.OutputBytes
			stats.rows += lstat.OutputRows
			stats.elapsed +=
				time.Duration(stat.EndTime-stat.StartTime) * time.Millisecond

			if stats.start.IsZero() {
				stats.start = time.Unix(stat.StartTime/1000, 0)
			} else {
				t := time.Unix(stat.StartTime/1000, 0)
				if stats.start.Sub(t) > 0 {
					stats.start = t
				}
			}

			if stats.finish.IsZero() {
				stats.finish = time.Unix(stat.EndTime/1000, 0)
			} else {
				t := time.Unix(stat.EndTime/1000, 0)
				if t.Sub(stats.finish) > 0 {
					stats.finish = t
				}
			}
		}
		// When the head job is processed reset the backoff since the loads
		// run in BQ in parallel.
		if loop {
			head = ""
			backoff = BaseBackoff
		}
	}

	fmt.Fprintf(os.Stderr, "%#v\n", stats)
}

type jobStats struct {
	// Number of files (sources) loaded.
	files int64
	// Bytes read from source (possibly compressed).
	bytesIn int64
	// Bytes loaded into BigQuery (uncompressed).
	bytesOut int64
	// Rows loaded into BigQuery.
	rows int64
	// Time taken to load source into table.
	elapsed time.Duration
	// Start time of the job.
	start time.Time
	// End time of the job.
	finish time.Time
}

func (s jobStats) GoString() string {
	return fmt.Sprintf("\n%d files loaded in %v (%v). Size: %.2fGB Rows: %d\n",
		s.files, s.finish.Sub(s.start), s.elapsed, float64(s.bytesOut)/GB,
		s.rows)
}
