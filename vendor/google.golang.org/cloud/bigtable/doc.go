/*
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Package bigtable is an API to Google Cloud Bigtable.

See https://cloud.google.com/bigtable/docs/ for general product documentation.

Setup and Credentials

Use NewClient or NewAdminClient to create a client that can be used to access
the data or admin APIs respectively. Both require credentials that have permission
to access the Cloud Bigtable API.

If your program is run on Google App Engine or Google Compute Engine, using the Application Default Credentials
(https://developers.google.com/accounts/docs/application-default-credentials)
is the simplest option. Those credentials will be used by default when NewClient or NewAdminClient are called.

To use alternate credentials, pass them to NewClient or NewAdminClient using cloud.WithTokenSource.
For instance, you can use service account credentials by visiting
https://cloud.google.com/console/project/MYPROJECT/apiui/credential,
creating a new OAuth "Client ID", storing the JSON key somewhere accessible, and writing
	jsonKey, err := ioutil.ReadFile(pathToKeyFile)
	...
	config, err := google.JWTConfigFromJSON(jsonKey, bigtable.Scope) // or bigtable.AdminScope, etc.
	...
	client, err := bigtable.NewClient(ctx, project, zone, cluster, cloud.WithTokenSource(config.TokenSource(ctx)))
	...
Here, `google` means the golang.org/x/oauth2/google package
and `cloud` means the google.golang.org/cloud package.

Reading

The principal way to read from a Bigtable is to use the ReadRows method on *Table.
A RowRange specifies a contiguous portion of a table. A Filter may be provided through
RowFilter to limit or transform the data that is returned.
	tbl := client.Open("mytable")
	...
	// Read all the rows starting with "com.google.",
	// but only fetch the columns in the "links" family.
	rr := bigtable.PrefixRange("com.google.")
	err := tbl.ReadRows(ctx, rr, func(r Row) bool {
		// do something with r
		return true // keep going
	}, bigtable.RowFilter(bigtable.FamilyFilter("links")))
	...

To read a single row, use the ReadRow helper method.
	r, err := tbl.ReadRow(ctx, "com.google.cloud") // "com.google.cloud" is the entire row key
	...

Writing

This API exposes two distinct forms of writing to a Bigtable: a Mutation and a ReadModifyWrite.
The former expresses idempotent operations.
The latter expresses non-idempotent operations and returns the new values of updated cells.
These operations are performed by creating a Mutation or ReadModifyWrite (with NewMutation or NewReadModifyWrite),
building up one or more operations on that, and then using the Apply or ApplyReadModifyWrite
methods on a Table.

For instance, to set a couple of cells in a table,
	tbl := client.Open("mytable")
	mut := bigtable.NewMutation()
	mut.Set("links", "maps.google.com", bigtable.Now(), []byte("1"))
	mut.Set("links", "golang.org", bigtable.Now(), []byte("1"))
	err := tbl.Apply(ctx, "com.google.cloud", mut)
	...

To increment an encoded value in one cell,
	tbl := client.Open("mytable")
	rmw := bigtable.NewReadModifyWrite()
	rmw.Increment("links", "golang.org", 12) // add 12 to the cell in column "links:golang.org"
	r, err := tbl.ApplyReadModifyWrite(ctx, "com.google.cloud", rmw)
	...
*/
package bigtable // import "google.golang.org/cloud/bigtable"

// Scope constants for authentication credentials.
// These should be used when using credential creation functions such as oauth.NewServiceAccountFromFile.
const (
	// Scope is the OAuth scope for Cloud Bigtable data operations.
	Scope = "https://www.googleapis.com/auth/bigtable.data"
	// ReadonlyScope is the OAuth scope for Cloud Bigtable read-only data operations.
	ReadonlyScope = "https://www.googleapis.com/auth/bigtable.readonly"

	// AdminScope is the OAuth scope for Cloud Bigtable table admin operations.
	AdminScope = "https://www.googleapis.com/auth/bigtable.admin.table"

	// ClusterAdminScope is the OAuth scope for Cloud Bigtable cluster admin operations.
	ClusterAdminScope = "https://www.googleapis.com/auth/bigtable.admin.cluster"
)

// clientUserAgent identifies the version of this package.
// It should be bumped upon significant changes only.
const clientUserAgent = "cbt-go/20150727"
