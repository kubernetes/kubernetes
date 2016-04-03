// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigquery

import bq "google.golang.org/api/bigquery/v2"

// GCSReference is a reference to one or more Google Cloud Storage objects, which together constitute
// an input or output to a BigQuery operation.
type GCSReference struct {
	uris []string

	// FieldDelimiter is the separator for fields in a CSV file, used when loading or exporting data.
	// The default is ",".
	FieldDelimiter string

	// The number of rows at the top of a CSV file that BigQuery will skip when loading the data.
	SkipLeadingRows int64

	// SourceFormat is the format of the GCS data to be loaded into BigQuery.
	// Allowed values are: CSV, JSON, DatastoreBackup.  The default is CSV.
	SourceFormat DataFormat
	// Only used when loading data.
	Encoding Encoding

	// Quote is the value used to quote data sections in a CSV file.
	// The default quotation character is the double quote ("), which is used if both Quote and ForceZeroQuote are unset.
	// To specify that no character should be interpreted as a quotation character, set ForceZeroQuote to true.
	// Only used when loading data.
	Quote          string
	ForceZeroQuote bool

	// DestinationFormat is the format to use when writing exported files.
	// Allowed values are: CSV, Avro, JSON.  The default is CSV.
	// CSV is not supported for tables with nested or repeated fields.
	DestinationFormat DataFormat
	// Only used when writing data.  Default is None.
	Compression Compression
}

func (gcs *GCSReference) implementsSource()      {}
func (gcs *GCSReference) implementsDestination() {}

// NewGCSReference constructs a reference to one or more Google Cloud Storage objects, which together constitute a data source or destination.
// In the simple case, a single URI in the form gs://bucket/object may refer to a single GCS object.
// Data may also be split into mutiple files, if multiple URIs or URIs containing wildcards are provided.
// Each URI may contain one '*' wildcard character, which (if present) must come after the bucket name.
// For more information about the treatment of wildcards and multiple URIs,
// see https://cloud.google.com/bigquery/exporting-data-from-bigquery#exportingmultiple
func (c *Client) NewGCSReference(uri ...string) *GCSReference {
	return &GCSReference{uris: uri}
}

type DataFormat string

const (
	CSV             DataFormat = "CSV"
	Avro            DataFormat = "AVRO"
	JSON            DataFormat = "NEWLINE_DELIMITED_JSON"
	DatastoreBackup DataFormat = "DATASTORE_BACKUP"
)

// Encoding specifies the character encoding of data to be loaded into BigQuery.
// See https://cloud.google.com/bigquery/docs/reference/v2/jobs#configuration.load.encoding
// for more details about how this is used.
type Encoding string

const (
	UTF_8      Encoding = "UTF-8"
	ISO_8859_1 Encoding = "ISO-8859-1"
)

// Compression is the type of compression to apply when writing data to Google Cloud Storage.
type Compression string

const (
	None Compression = "NONE"
	Gzip Compression = "GZIP"
)

func (gcs *GCSReference) customizeLoadSrc(conf *bq.JobConfigurationLoad, projectID string) {
	conf.SourceUris = gcs.uris
	conf.SkipLeadingRows = gcs.SkipLeadingRows
	conf.SourceFormat = string(gcs.SourceFormat)
	conf.Encoding = string(gcs.Encoding)
	conf.FieldDelimiter = gcs.FieldDelimiter

	if gcs.ForceZeroQuote {
		quote := ""
		conf.Quote = &quote
	} else if gcs.Quote != "" {
		conf.Quote = &gcs.Quote
	}
}

func (gcs *GCSReference) customizeExtractDst(conf *bq.JobConfigurationExtract, projectID string) {
	conf.DestinationUris = gcs.uris
	conf.Compression = string(gcs.Compression)
	conf.DestinationFormat = string(gcs.DestinationFormat)
	conf.FieldDelimiter = gcs.FieldDelimiter
}
