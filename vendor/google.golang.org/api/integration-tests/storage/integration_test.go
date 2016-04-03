// +build integration

package storage

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"testing"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/googleapi"
	storage "google.golang.org/api/storage/v1"
)

type object struct {
	name, contents string
}

var (
	projectID string
	bucket    string
	objects   = []object{
		{"obj1", testContents},
		{"obj2", testContents},
		{"obj/with/slashes", testContents},
		{"resumable", testContents},
		{"large", strings.Repeat("a", 514)}, // larger than the first section of content that is sniffed by ContentSniffer.
	}
	aclObjects = []string{"acl1", "acl2"}
	copyObj    = "copy-object"
)

const (
	envProject    = "GCLOUD_TESTS_GOLANG_PROJECT_ID"
	envPrivateKey = "GCLOUD_TESTS_GOLANG_KEY"
	// NOTE that running this test on a bucket deletes ALL contents of the bucket!
	envBucket    = "GCLOUD_TESTS_GOLANG_DESTRUCTIVE_TEST_BUCKET_NAME"
	testContents = "some text that will be saved to a bucket object"
)

func verifyAcls(obj *storage.Object, wantDomainRole, wantAllUsersRole string) (err error) {
	var gotDomainRole, gotAllUsersRole string
	for _, acl := range obj.Acl {
		if acl.Entity == "domain-google.com" {
			gotDomainRole = acl.Role
		}
		if acl.Entity == "allUsers" {
			gotAllUsersRole = acl.Role
		}
	}
	if gotDomainRole != wantDomainRole {
		err = fmt.Errorf("domain-google.com role = %q; want %q", gotDomainRole, wantDomainRole)
	}
	if gotAllUsersRole != wantAllUsersRole {
		err = fmt.Errorf("allUsers role = %q; want %q; %v", gotAllUsersRole, wantAllUsersRole, err)
	}
	return err
}

// TODO(gmlewis): Move this to a common location.
func tokenSource(ctx context.Context, scopes ...string) (oauth2.TokenSource, error) {
	keyFile := os.Getenv(envPrivateKey)
	if keyFile == "" {
		return nil, errors.New(envPrivateKey + " not set")
	}
	jsonKey, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, fmt.Errorf("unable to read %q: %v", keyFile, err)
	}
	conf, err := google.JWTConfigFromJSON(jsonKey, scopes...)
	if err != nil {
		return nil, fmt.Errorf("google.JWTConfigFromJSON: %v", err)
	}
	return conf.TokenSource(ctx), nil
}

const defaultType = "text/plain; charset=utf-8"

// writeObject writes some data and default metadata to the specified object.
// Resumable upload is used if resumable is true.
// The written data is returned.
func writeObject(s *storage.Service, bucket, obj string, resumable bool, contents string) error {
	o := &storage.Object{
		Bucket:          bucket,
		Name:            obj,
		ContentType:     defaultType,
		ContentEncoding: "utf-8",
		ContentLanguage: "en",
		Metadata:        map[string]string{"foo": "bar"},
	}
	f := strings.NewReader(contents)
	insert := s.Objects.Insert(bucket, o)
	if resumable {
		insert.ResumableMedia(context.Background(), f, int64(len(contents)), defaultType)
	} else {
		insert.Media(f)
	}
	_, err := insert.Do()
	return err
}

func checkMetadata(t *testing.T, s *storage.Service, bucket, obj string) {
	o, err := s.Objects.Get(bucket, obj).Do()
	if err != nil {
		t.Error(err)
	}
	if got, want := o.Name, obj; got != want {
		t.Errorf("name of %q = %q; want %q", obj, got, want)
	}
	if got, want := o.ContentType, defaultType; got != want {
		t.Errorf("contentType of %q = %q; want %q", obj, got, want)
	}
	if got, want := o.Metadata["foo"], "bar"; got != want {
		t.Errorf("metadata entry foo of %q = %q; want %q", obj, got, want)
	}
}

func TestFunctions(t *testing.T) {
	if projectID = os.Getenv(envProject); projectID == "" {
		t.Fatalf("no project ID specified")
	}
	if bucket = os.Getenv(envBucket); bucket == "" {
		t.Fatalf("no bucket specified")
	}

	ctx := context.Background()
	ts, err := tokenSource(ctx, storage.DevstorageFullControlScope)
	if err != nil {
		t.Fatal(err)
	}
	client := oauth2.NewClient(ctx, ts)
	s, err := storage.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}

	cleanup(t, s)
	defer cleanup(t, s)

	t.Logf("Listing buckets for project %q", projectID)
	var numBuckets int
	pageToken := ""
	for {
		call := s.Buckets.List(projectID)
		if pageToken != "" {
			call.PageToken(pageToken)
		}
		resp, err := call.Do()
		if err != nil {
			t.Fatalf("unable to list buckets for project %q: %v", projectID, err)
		}
		numBuckets += len(resp.Items)
		if pageToken = resp.NextPageToken; pageToken == "" {
			break
		}
	}
	if numBuckets == 0 {
		t.Fatalf("no buckets found for project %q", projectID)
	}

	for _, obj := range objects {
		t.Logf("Writing %q", obj.name)
		// TODO(mcgreevy): stop relying on "resumable" name to determine whether to
		// do an resumable upload.
		err := writeObject(s, bucket, obj.name, obj.name == "resumable", obj.contents)
		if err != nil {
			t.Fatalf("unable to insert object %q: %v", obj.name, err)
		}
	}

	for _, obj := range objects {
		t.Logf("Reading %q", obj.name)
		resp, err := s.Objects.Get(bucket, obj.name).Download()
		if err != nil {
			t.Fatalf("unable to get object %q: %v", obj.name, err)
		}
		slurp, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("unable to read response body %q: %v", obj.name, err)
		}
		resp.Body.Close()
		if got, want := string(slurp), obj.contents; got != want {
			t.Errorf("contents of %q = %q; want %q", obj.name, got, want)
		}
	}

	name := "obj-not-exists"
	if _, err := s.Objects.Get(bucket, name).Download(); !isError(err, http.StatusNotFound) {
		t.Errorf("object %q should not exist, err = %v", name, err)
	} else {
		t.Log("Successfully tested StatusNotFound.")
	}

	for _, obj := range objects {
		t.Logf("Checking %q metadata", obj.name)
		checkMetadata(t, s, bucket, obj.name)
	}

	name = objects[0].name

	t.Logf("Rewriting %q to %q", name, copyObj)
	copy, err := s.Objects.Rewrite(bucket, name, bucket, copyObj, nil).Do()
	if err != nil {
		t.Errorf("unable to rewrite object %q to %q: %v", name, copyObj, err)
	}
	if copy.Resource.Name != copyObj {
		t.Errorf("copy object's name = %q; want %q", copy.Resource.Name, copyObj)
	}
	if copy.Resource.Bucket != bucket {
		t.Errorf("copy object's bucket = %q; want %q", copy.Resource.Bucket, bucket)
	}

	// Note that arrays such as ACLs below are completely overwritten using Patch
	// semantics, so these must be updated in a read-modify-write sequence of operations.
	// See https://cloud.google.com/storage/docs/json_api/v1/how-tos/performance#patch-semantics
	// for more details.
	t.Logf("Updating attributes of %q", name)
	obj, err := s.Objects.Get(bucket, name).Projection("full").Fields("acl").Do()
	if err != nil {
		t.Errorf("Objects.Get(%q, %q): %v", bucket, name, err)
	}
	if err := verifyAcls(obj, "", ""); err != nil {
		t.Errorf("before update ACLs: %v", err)
	}
	obj.ContentType = "text/html"
	for _, entity := range []string{"domain-google.com", "allUsers"} {
		obj.Acl = append(obj.Acl, &storage.ObjectAccessControl{Entity: entity, Role: "READER"})
	}
	updated, err := s.Objects.Patch(bucket, name, obj).Projection("full").Fields("contentType", "acl").Do()
	if err != nil {
		t.Errorf("Objects.Patch(%q, %q, %#v) failed with %v", bucket, name, obj, err)
	}
	if want := "text/html"; updated.ContentType != want {
		t.Errorf("updated.ContentType == %q; want %q", updated.ContentType, want)
	}
	if err := verifyAcls(updated, "READER", "READER"); err != nil {
		t.Errorf("after update ACLs: %v", err)
	}

	t.Log("Testing checksums")
	checksumCases := []struct {
		name     string
		contents string
		size     uint64
		md5      string
		crc32c   uint32
	}{
		{
			name:     "checksum-object",
			contents: "helloworld",
			size:     10,
			md5:      "fc5e038d38a57032085441e7fe7010b0",
			crc32c:   1456190592,
		},
		{
			name:     "zero-object",
			contents: "",
			size:     0,
			md5:      "d41d8cd98f00b204e9800998ecf8427e",
			crc32c:   0,
		},
	}
	for _, c := range checksumCases {
		f := strings.NewReader(c.contents)
		o := &storage.Object{
			Bucket:          bucket,
			Name:            c.name,
			ContentType:     defaultType,
			ContentEncoding: "utf-8",
			ContentLanguage: "en",
		}
		obj, err := s.Objects.Insert(bucket, o).Media(f).Do()
		if err != nil {
			t.Fatalf("unable to insert object %q: %v", obj, err)
		}
		if got, want := obj.Size, c.size; got != want {
			t.Errorf("object %q size = %v; want %v", c.name, got, want)
		}
		md5, err := base64.StdEncoding.DecodeString(obj.Md5Hash)
		if err != nil {
			t.Errorf("object %q base64 decode of MD5 %q: %v", c.name, obj.Md5Hash, err)
		}
		if got, want := fmt.Sprintf("%x", md5), c.md5; got != want {
			t.Errorf("object %q MD5 = %q; want %q", c.name, got, want)
		}
		var crc32c uint32
		d, err := base64.StdEncoding.DecodeString(obj.Crc32c)
		if err != nil {
			t.Errorf("object %q base64 decode of CRC32 %q: %v", c.name, obj.Crc32c, err)
		}
		if err == nil && len(d) == 4 {
			crc32c = uint32(d[0])<<24 + uint32(d[1])<<16 + uint32(d[2])<<8 + uint32(d[3])
		}
		if got, want := crc32c, c.crc32c; got != want {
			t.Errorf("object %q CRC32C = %v; want %v", c.name, got, want)
		}
	}
}

// cleanup destroys ALL objects in the bucket!
func cleanup(t *testing.T, s *storage.Service) {
	var pageToken string
	for {
		call := s.Objects.List(bucket)
		if pageToken != "" {
			call.PageToken(pageToken)
		}
		resp, err := call.Do()
		if err != nil {
			t.Fatalf("unable to list bucket %q: %v", bucket, err)
		}
		for _, obj := range resp.Items {
			t.Logf("Cleanup deletion of %q", obj.Name)
			if err := s.Objects.Delete(bucket, obj.Name).Do(); err != nil {
				t.Fatalf("unable to delete %q: %v", obj.Name, err)
			}
			if _, err := s.Objects.Get(bucket, obj.Name).Download(); !isError(err, http.StatusNotFound) {
				t.Errorf("object %q should not exist, err = %v", obj.Name, err)
			} else {
				t.Logf("Successfully deleted %q.", obj.Name)
			}
		}
		if pageToken = resp.NextPageToken; pageToken == "" {
			break
		}
	}
}

func isError(err error, code int) bool {
	if err == nil {
		return false
	}
	ae, ok := err.(*googleapi.Error)
	return ok && ae.Code == code
}
