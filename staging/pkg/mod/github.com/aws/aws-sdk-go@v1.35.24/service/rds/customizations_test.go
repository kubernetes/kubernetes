// +build go1.9

package rds

import (
	"fmt"
	"io/ioutil"
	"net/url"
	"regexp"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

func TestCopyDBSnapshotNoPanic(t *testing.T) {
	svc := New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})

	f := func() {
		// Doesn't panic on nil input
		req, _ := svc.CopyDBSnapshotRequest(nil)
		req.Sign()
	}
	if paniced, p := awstesting.DidPanic(f); paniced {
		t.Errorf("expect no panic, got %v", p)
	}
}

func TestPresignCrossRegionRequest(t *testing.T) {
	const targetRegion = "us-west-2"

	svc := New(unit.Session, &aws.Config{Region: aws.String(targetRegion)})

	const regexPattern = `^https://rds.us-west-1\.amazonaws\.com/\?Action=%s.+?DestinationRegion=%s.+`

	cases := map[string]struct {
		Req    *request.Request
		Assert func(*testing.T, string)
	}{
		opCopyDBSnapshot: {
			Req: func() *request.Request {
				req, _ := svc.CopyDBSnapshotRequest(&CopyDBSnapshotInput{
					SourceRegion:               aws.String("us-west-1"),
					SourceDBSnapshotIdentifier: aws.String("foo"),
					TargetDBSnapshotIdentifier: aws.String("bar"),
				})
				return req
			}(),
			Assert: assertAsRegexMatch(fmt.Sprintf(regexPattern,
				opCopyDBSnapshot, targetRegion)),
		},
		opCreateDBInstanceReadReplica: {
			Req: func() *request.Request {
				req, _ := svc.CreateDBInstanceReadReplicaRequest(
					&CreateDBInstanceReadReplicaInput{
						SourceRegion:               aws.String("us-west-1"),
						SourceDBInstanceIdentifier: aws.String("foo"),
						DBInstanceIdentifier:       aws.String("bar"),
					})
				return req
			}(),
			Assert: assertAsRegexMatch(fmt.Sprintf(regexPattern,
				opCreateDBInstanceReadReplica, targetRegion)),
		},
		opCopyDBClusterSnapshot: {
			Req: func() *request.Request {
				req, _ := svc.CopyDBClusterSnapshotRequest(
					&CopyDBClusterSnapshotInput{
						SourceRegion:                      aws.String("us-west-1"),
						SourceDBClusterSnapshotIdentifier: aws.String("foo"),
						TargetDBClusterSnapshotIdentifier: aws.String("bar"),
					})
				return req
			}(),
			Assert: assertAsRegexMatch(fmt.Sprintf(regexPattern,
				opCopyDBClusterSnapshot, targetRegion)),
		},
		opCreateDBCluster: {
			Req: func() *request.Request {
				req, _ := svc.CreateDBClusterRequest(
					&CreateDBClusterInput{
						SourceRegion:        aws.String("us-west-1"),
						DBClusterIdentifier: aws.String("foo"),
						Engine:              aws.String("bar"),
					})
				return req
			}(),
			Assert: assertAsRegexMatch(fmt.Sprintf(regexPattern,
				opCreateDBCluster, targetRegion)),
		},
		opCopyDBSnapshot + " same region": {
			Req: func() *request.Request {
				req, _ := svc.CopyDBSnapshotRequest(&CopyDBSnapshotInput{
					SourceRegion:               aws.String("us-west-2"),
					SourceDBSnapshotIdentifier: aws.String("foo"),
					TargetDBSnapshotIdentifier: aws.String("bar"),
				})
				return req
			}(),
			Assert: assertAsEmpty(),
		},
		opCreateDBInstanceReadReplica + " same region": {
			Req: func() *request.Request {
				req, _ := svc.CreateDBInstanceReadReplicaRequest(&CreateDBInstanceReadReplicaInput{
					SourceRegion:               aws.String("us-west-2"),
					SourceDBInstanceIdentifier: aws.String("foo"),
					DBInstanceIdentifier:       aws.String("bar"),
				})
				return req
			}(),
			Assert: assertAsEmpty(),
		},
		opCopyDBClusterSnapshot + " same region": {
			Req: func() *request.Request {
				req, _ := svc.CopyDBClusterSnapshotRequest(
					&CopyDBClusterSnapshotInput{
						SourceRegion:                      aws.String("us-west-2"),
						SourceDBClusterSnapshotIdentifier: aws.String("foo"),
						TargetDBClusterSnapshotIdentifier: aws.String("bar"),
					})
				return req
			}(),
			Assert: assertAsEmpty(),
		},
		opCreateDBCluster + " same region": {
			Req: func() *request.Request {
				req, _ := svc.CreateDBClusterRequest(
					&CreateDBClusterInput{
						SourceRegion:        aws.String("us-west-2"),
						DBClusterIdentifier: aws.String("foo"),
						Engine:              aws.String("bar"),
					})
				return req
			}(),
			Assert: assertAsEmpty(),
		},
		opCopyDBSnapshot + " presignURL set": {
			Req: func() *request.Request {
				req, _ := svc.CopyDBSnapshotRequest(&CopyDBSnapshotInput{
					SourceRegion:               aws.String("us-west-1"),
					SourceDBSnapshotIdentifier: aws.String("foo"),
					TargetDBSnapshotIdentifier: aws.String("bar"),
					PreSignedUrl:               aws.String("mockPresignedURL"),
				})
				return req
			}(),
			Assert: assertAsEqual("mockPresignedURL"),
		},
		opCreateDBInstanceReadReplica + " presignURL set": {
			Req: func() *request.Request {
				req, _ := svc.CreateDBInstanceReadReplicaRequest(&CreateDBInstanceReadReplicaInput{
					SourceRegion:               aws.String("us-west-1"),
					SourceDBInstanceIdentifier: aws.String("foo"),
					DBInstanceIdentifier:       aws.String("bar"),
					PreSignedUrl:               aws.String("mockPresignedURL"),
				})
				return req
			}(),
			Assert: assertAsEqual("mockPresignedURL"),
		},
		opCopyDBClusterSnapshot + " presignURL set": {
			Req: func() *request.Request {
				req, _ := svc.CopyDBClusterSnapshotRequest(
					&CopyDBClusterSnapshotInput{
						SourceRegion:                      aws.String("us-west-1"),
						SourceDBClusterSnapshotIdentifier: aws.String("foo"),
						TargetDBClusterSnapshotIdentifier: aws.String("bar"),
						PreSignedUrl:                      aws.String("mockPresignedURL"),
					})
				return req
			}(),
			Assert: assertAsEqual("mockPresignedURL"),
		},
		opCreateDBCluster + " presignURL set": {
			Req: func() *request.Request {
				req, _ := svc.CreateDBClusterRequest(
					&CreateDBClusterInput{
						SourceRegion:        aws.String("us-west-1"),
						DBClusterIdentifier: aws.String("foo"),
						Engine:              aws.String("bar"),
						PreSignedUrl:        aws.String("mockPresignedURL"),
					})
				return req
			}(),
			Assert: assertAsEqual("mockPresignedURL"),
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			if err := c.Req.Sign(); err != nil {
				t.Fatalf("expect no error, got %v", err)
			}
			b, _ := ioutil.ReadAll(c.Req.HTTPRequest.Body)
			q, _ := url.ParseQuery(string(b))

			u, _ := url.QueryUnescape(q.Get("PreSignedUrl"))

			c.Assert(t, u)
		})
	}
}

func TestPresignWithSourceNotSet(t *testing.T) {
	reqs := map[string]*request.Request{}
	svc := New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})

	reqs[opCopyDBSnapshot], _ = svc.CopyDBSnapshotRequest(&CopyDBSnapshotInput{
		SourceDBSnapshotIdentifier: aws.String("foo"),
		TargetDBSnapshotIdentifier: aws.String("bar"),
	})

	for _, req := range reqs {
		_, err := req.Presign(5 * time.Minute)
		if err != nil {
			t.Fatal(err)
		}
	}
}

func assertAsRegexMatch(exp string) func(*testing.T, string) {
	return func(t *testing.T, v string) {
		t.Helper()

		if re, a := regexp.MustCompile(exp), v; !re.MatchString(a) {
			t.Errorf("expect %s to match %s", re, a)
		}
	}
}

func assertAsEmpty() func(*testing.T, string) {
	return func(t *testing.T, v string) {
		t.Helper()

		if len(v) != 0 {
			t.Errorf("expect empty, got %v", v)
		}
	}
}

func assertAsEqual(expect string) func(*testing.T, string) {
	return func(t *testing.T, v string) {
		t.Helper()

		if e, a := expect, v; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
	}
}
