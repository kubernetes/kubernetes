// Copyright 2014 Google Inc. All Rights Reserved.
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

//[START sample]
// Package gcsdemo is an example App Engine app using the Google Cloud Storage API.
//
// NOTE: the cloud.google.com/go/storage package is not compatible with
// dev_appserver.py, so this example will not work in a local development
// environment.
package gcsdemo

//[START imports]
import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"

	"cloud.google.com/go/storage"
	"golang.org/x/net/context"
	"google.golang.org/appengine"
	"google.golang.org/appengine/file"
	"google.golang.org/appengine/log"
)

//[END imports]

// bucket is a local cache of the app's default bucket name.
var bucket string // or: var bucket = "<your-app-id>.appspot.com"

func init() {
	http.HandleFunc("/", handler)
}

// demo struct holds information needed to run the various demo functions.
type demo struct {
	bucket *storage.BucketHandle
	client *storage.Client

	w   io.Writer
	ctx context.Context
	// cleanUp is a list of filenames that need cleaning up at the end of the demo.
	cleanUp []string
	// failed indicates that one or more of the demo steps failed.
	failed bool
}

func (d *demo) errorf(format string, args ...interface{}) {
	d.failed = true
	fmt.Fprintln(d.w, fmt.Sprintf(format, args...))
	log.Errorf(d.ctx, format, args...)
}

// handler is the main demo entry point that calls the GCS operations.
func handler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	if appengine.IsDevAppServer() {
		http.Error(w, "This example does not work with dev_appserver.py", http.StatusNotImplemented)
	}

	//[START get_default_bucket]
	ctx := appengine.NewContext(r)
	if bucket == "" {
		var err error
		if bucket, err = file.DefaultBucketName(ctx); err != nil {
			log.Errorf(ctx, "failed to get default GCS bucket name: %v", err)
			return
		}
	}
	//[END get_default_bucket]

	client, err := storage.NewClient(ctx)
	if err != nil {
		log.Errorf(ctx, "failed to create client: %v", err)
		return
	}
	defer client.Close()

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	fmt.Fprintf(w, "Demo GCS Application running from Version: %v\n", appengine.VersionID(ctx))
	fmt.Fprintf(w, "Using bucket name: %v\n\n", bucket)

	buf := &bytes.Buffer{}
	d := &demo{
		w:      buf,
		ctx:    ctx,
		client: client,
		bucket: client.Bucket(bucket),
	}

	n := "demo-testfile-go"
	d.createFile(n)
	d.readFile(n)
	d.copyFile(n)
	d.statFile(n)
	d.createListFiles()
	d.listBucket()
	d.listBucketDirMode()
	d.defaultACL()
	d.putDefaultACLRule()
	d.deleteDefaultACLRule()
	d.bucketACL()
	d.putBucketACLRule()
	d.deleteBucketACLRule()
	d.acl(n)
	d.putACLRule(n)
	d.deleteACLRule(n)
	d.deleteFiles()

	if d.failed {
		w.WriteHeader(http.StatusInternalServerError)
		buf.WriteTo(w)
		fmt.Fprintf(w, "\nDemo failed.\n")
	} else {
		w.WriteHeader(http.StatusOK)
		buf.WriteTo(w)
		fmt.Fprintf(w, "\nDemo succeeded.\n")
	}
}

//[START write]
// createFile creates a file in Google Cloud Storage.
func (d *demo) createFile(fileName string) {
	fmt.Fprintf(d.w, "Creating file /%v/%v\n", bucket, fileName)

	wc := d.bucket.Object(fileName).NewWriter(d.ctx)
	wc.ContentType = "text/plain"
	wc.Metadata = map[string]string{
		"x-goog-meta-foo": "foo",
		"x-goog-meta-bar": "bar",
	}
	d.cleanUp = append(d.cleanUp, fileName)

	if _, err := wc.Write([]byte("abcde\n")); err != nil {
		d.errorf("createFile: unable to write data to bucket %q, file %q: %v", bucket, fileName, err)
		return
	}
	if _, err := wc.Write([]byte(strings.Repeat("f", 1024*4) + "\n")); err != nil {
		d.errorf("createFile: unable to write data to bucket %q, file %q: %v", bucket, fileName, err)
		return
	}
	if err := wc.Close(); err != nil {
		d.errorf("createFile: unable to close bucket %q, file %q: %v", bucket, fileName, err)
		return
	}
}

//[END write]

//[START read]
// readFile reads the named file in Google Cloud Storage.
func (d *demo) readFile(fileName string) {
	io.WriteString(d.w, "\nAbbreviated file content (first line and last 1K):\n")

	rc, err := d.bucket.Object(fileName).NewReader(d.ctx)
	if err != nil {
		d.errorf("readFile: unable to open file from bucket %q, file %q: %v", bucket, fileName, err)
		return
	}
	defer rc.Close()
	slurp, err := ioutil.ReadAll(rc)
	if err != nil {
		d.errorf("readFile: unable to read data from bucket %q, file %q: %v", bucket, fileName, err)
		return
	}

	fmt.Fprintf(d.w, "%s\n", bytes.SplitN(slurp, []byte("\n"), 2)[0])
	if len(slurp) > 1024 {
		fmt.Fprintf(d.w, "...%s\n", slurp[len(slurp)-1024:])
	} else {
		fmt.Fprintf(d.w, "%s\n", slurp)
	}
}

//[END read]

//[START copy]
// copyFile copies a file in Google Cloud Storage.
func (d *demo) copyFile(fileName string) {
	copyName := fileName + "-copy"
	fmt.Fprintf(d.w, "Copying file /%v/%v to /%v/%v:\n", bucket, fileName, bucket, copyName)

	obj, err := d.bucket.Object(fileName).CopyTo(d.ctx, d.bucket.Object(copyName), nil)
	if err != nil {
		d.errorf("copyFile: unable to copy /%v/%v to bucket %q, file %q: %v", bucket, fileName, bucket, copyName, err)
		return
	}
	d.cleanUp = append(d.cleanUp, copyName)

	d.dumpStats(obj)
}

//[END copy]

func (d *demo) dumpStats(obj *storage.ObjectAttrs) {
	fmt.Fprintf(d.w, "(filename: /%v/%v, ", obj.Bucket, obj.Name)
	fmt.Fprintf(d.w, "ContentType: %q, ", obj.ContentType)
	fmt.Fprintf(d.w, "ACL: %#v, ", obj.ACL)
	fmt.Fprintf(d.w, "Owner: %v, ", obj.Owner)
	fmt.Fprintf(d.w, "ContentEncoding: %q, ", obj.ContentEncoding)
	fmt.Fprintf(d.w, "Size: %v, ", obj.Size)
	fmt.Fprintf(d.w, "MD5: %q, ", obj.MD5)
	fmt.Fprintf(d.w, "CRC32C: %q, ", obj.CRC32C)
	fmt.Fprintf(d.w, "Metadata: %#v, ", obj.Metadata)
	fmt.Fprintf(d.w, "MediaLink: %q, ", obj.MediaLink)
	fmt.Fprintf(d.w, "StorageClass: %q, ", obj.StorageClass)
	if !obj.Deleted.IsZero() {
		fmt.Fprintf(d.w, "Deleted: %v, ", obj.Deleted)
	}
	fmt.Fprintf(d.w, "Updated: %v)\n", obj.Updated)
}

//[START file_metadata]
// statFile reads the stats of the named file in Google Cloud Storage.
func (d *demo) statFile(fileName string) {
	io.WriteString(d.w, "\nFile stat:\n")

	obj, err := d.bucket.Object(fileName).Attrs(d.ctx)
	if err != nil {
		d.errorf("statFile: unable to stat file from bucket %q, file %q: %v", bucket, fileName, err)
		return
	}

	d.dumpStats(obj)
}

//[END file_metadata]

// createListFiles creates files that will be used by listBucket.
func (d *demo) createListFiles() {
	io.WriteString(d.w, "\nCreating more files for listbucket...\n")
	for _, n := range []string{"foo1", "foo2", "bar", "bar/1", "bar/2", "boo/"} {
		d.createFile(n)
	}
}

//[START list_bucket]
// listBucket lists the contents of a bucket in Google Cloud Storage.
func (d *demo) listBucket() {
	io.WriteString(d.w, "\nListbucket result:\n")

	query := &storage.Query{Prefix: "foo"}
	for query != nil {
		objs, err := d.bucket.List(d.ctx, query)
		if err != nil {
			d.errorf("listBucket: unable to list bucket %q: %v", bucket, err)
			return
		}
		query = objs.Next

		for _, obj := range objs.Results {
			d.dumpStats(obj)
		}
	}
}

//[END list_bucket]

func (d *demo) listDir(name, indent string) {
	query := &storage.Query{Prefix: name, Delimiter: "/"}
	for query != nil {
		objs, err := d.bucket.List(d.ctx, query)
		if err != nil {
			d.errorf("listBucketDirMode: unable to list bucket %q: %v", bucket, err)
			return
		}
		query = objs.Next

		for _, obj := range objs.Results {
			fmt.Fprint(d.w, indent)
			d.dumpStats(obj)
		}
		for _, dir := range objs.Prefixes {
			fmt.Fprintf(d.w, "%v(directory: /%v/%v)\n", indent, bucket, dir)
			d.listDir(dir, indent+"  ")
		}
	}
}

// listBucketDirMode lists the contents of a bucket in dir mode in Google Cloud Storage.
func (d *demo) listBucketDirMode() {
	io.WriteString(d.w, "\nListbucket directory mode result:\n")
	d.listDir("b", "")
}

// dumpDefaultACL prints out the default object ACL for this bucket.
func (d *demo) dumpDefaultACL() {
	acl, err := d.bucket.ACL().List(d.ctx)
	if err != nil {
		d.errorf("defaultACL: unable to list default object ACL for bucket %q: %v", bucket, err)
		return
	}
	for _, v := range acl {
		fmt.Fprintf(d.w, "Scope: %q, Permission: %q\n", v.Entity, v.Role)
	}
}

// defaultACL displays the default object ACL for this bucket.
func (d *demo) defaultACL() {
	io.WriteString(d.w, "\nDefault object ACL:\n")
	d.dumpDefaultACL()
}

// putDefaultACLRule adds the "allUsers" default object ACL rule for this bucket.
func (d *demo) putDefaultACLRule() {
	io.WriteString(d.w, "\nPut Default object ACL Rule:\n")
	err := d.bucket.DefaultObjectACL().Set(d.ctx, storage.AllUsers, storage.RoleReader)
	if err != nil {
		d.errorf("putDefaultACLRule: unable to save default object ACL rule for bucket %q: %v", bucket, err)
		return
	}
	d.dumpDefaultACL()
}

// deleteDefaultACLRule deleted the "allUsers" default object ACL rule for this bucket.
func (d *demo) deleteDefaultACLRule() {
	io.WriteString(d.w, "\nDelete Default object ACL Rule:\n")
	err := d.bucket.DefaultObjectACL().Delete(d.ctx, storage.AllUsers)
	if err != nil {
		d.errorf("deleteDefaultACLRule: unable to delete default object ACL rule for bucket %q: %v", bucket, err)
		return
	}
	d.dumpDefaultACL()
}

// dumpBucketACL prints out the bucket ACL.
func (d *demo) dumpBucketACL() {
	acl, err := d.bucket.ACL().List(d.ctx)
	if err != nil {
		d.errorf("dumpBucketACL: unable to list bucket ACL for bucket %q: %v", bucket, err)
		return
	}
	for _, v := range acl {
		fmt.Fprintf(d.w, "Scope: %q, Permission: %q\n", v.Entity, v.Role)
	}
}

// bucketACL displays the bucket ACL for this bucket.
func (d *demo) bucketACL() {
	io.WriteString(d.w, "\nBucket ACL:\n")
	d.dumpBucketACL()
}

// putBucketACLRule adds the "allUsers" bucket ACL rule for this bucket.
func (d *demo) putBucketACLRule() {
	io.WriteString(d.w, "\nPut Bucket ACL Rule:\n")
	err := d.bucket.ACL().Set(d.ctx, storage.AllUsers, storage.RoleReader)
	if err != nil {
		d.errorf("putBucketACLRule: unable to save bucket ACL rule for bucket %q: %v", bucket, err)
		return
	}
	d.dumpBucketACL()
}

// deleteBucketACLRule deleted the "allUsers" bucket ACL rule for this bucket.
func (d *demo) deleteBucketACLRule() {
	io.WriteString(d.w, "\nDelete Bucket ACL Rule:\n")
	err := d.bucket.ACL().Delete(d.ctx, storage.AllUsers)
	if err != nil {
		d.errorf("deleteBucketACLRule: unable to delete bucket ACL rule for bucket %q: %v", bucket, err)
		return
	}
	d.dumpBucketACL()
}

// dumpACL prints out the ACL of the named file.
func (d *demo) dumpACL(fileName string) {
	acl, err := d.bucket.Object(fileName).ACL().List(d.ctx)
	if err != nil {
		d.errorf("dumpACL: unable to list file ACL for bucket %q, file %q: %v", bucket, fileName, err)
		return
	}
	for _, v := range acl {
		fmt.Fprintf(d.w, "Scope: %q, Permission: %q\n", v.Entity, v.Role)
	}
}

// acl displays the ACL for the named file.
func (d *demo) acl(fileName string) {
	fmt.Fprintf(d.w, "\nACL for file %v:\n", fileName)
	d.dumpACL(fileName)
}

// putACLRule adds the "allUsers" ACL rule for the named file.
func (d *demo) putACLRule(fileName string) {
	fmt.Fprintf(d.w, "\nPut ACL rule for file %v:\n", fileName)
	err := d.bucket.Object(fileName).ACL().Set(d.ctx, storage.AllUsers, storage.RoleReader)
	if err != nil {
		d.errorf("putACLRule: unable to save ACL rule for bucket %q, file %q: %v", bucket, fileName, err)
		return
	}
	d.dumpACL(fileName)
}

// deleteACLRule deleted the "allUsers" ACL rule for the named file.
func (d *demo) deleteACLRule(fileName string) {
	fmt.Fprintf(d.w, "\nDelete ACL rule for file %v:\n", fileName)
	err := d.bucket.Object(fileName).ACL().Delete(d.ctx, storage.AllUsers)
	if err != nil {
		d.errorf("deleteACLRule: unable to delete ACL rule for bucket %q, file %q: %v", bucket, fileName, err)
		return
	}
	d.dumpACL(fileName)
}

// deleteFiles deletes all the temporary files from a bucket created by this demo.
func (d *demo) deleteFiles() {
	io.WriteString(d.w, "\nDeleting files...\n")
	for _, v := range d.cleanUp {
		fmt.Fprintf(d.w, "Deleting file %v\n", v)
		if err := d.bucket.Object(v).Delete(d.ctx); err != nil {
			d.errorf("deleteFiles: unable to delete bucket %q, file %q: %v", bucket, v, err)
			return
		}
	}
}

//[END sample]
