// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package log provides the means of querying an application's logs from
within an App Engine application.

Example:
	c := appengine.NewContext(r)
	query := &log.Query{
		AppLogs:  true,
		Versions: []string{"1"},
	}

	for results := query.Run(c); ; {
		record, err := results.Next()
		if err == log.Done {
			log.Infof(c, "Done processing results")
			break
		}
		if err != nil {
			log.Errorf(c, "Failed to retrieve next log: %v", err)
			break
		}
		log.Infof(c, "Saw record %v", record)
	}
*/
package log // import "google.golang.org/appengine/log"

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/log"
)

// Query defines a logs query.
type Query struct {
	// Start time specifies the earliest log to return (inclusive).
	StartTime time.Time

	// End time specifies the latest log to return (exclusive).
	EndTime time.Time

	// Offset specifies a position within the log stream to resume reading from,
	// and should come from a previously returned Record's field of the same name.
	Offset []byte

	// Incomplete controls whether active (incomplete) requests should be included.
	Incomplete bool

	// AppLogs indicates if application-level logs should be included.
	AppLogs bool

	// ApplyMinLevel indicates if MinLevel should be used to filter results.
	ApplyMinLevel bool

	// If ApplyMinLevel is true, only logs for requests with at least one
	// application log of MinLevel or higher will be returned.
	MinLevel int

	// Versions is the major version IDs whose logs should be retrieved.
	// Logs for specific modules can be retrieved by the specifying versions
	// in the form "module:version"; the default module is used if no module
	// is specified.
	Versions []string

	// A list of requests to search for instead of a time-based scan. Cannot be
	// combined with filtering options such as StartTime, EndTime, Offset,
	// Incomplete, ApplyMinLevel, or Versions.
	RequestIDs []string
}

// AppLog represents a single application-level log.
type AppLog struct {
	Time    time.Time
	Level   int
	Message string
}

// Record contains all the information for a single web request.
type Record struct {
	AppID            string
	ModuleID         string
	VersionID        string
	RequestID        []byte
	IP               string
	Nickname         string
	AppEngineRelease string

	// The time when this request started.
	StartTime time.Time

	// The time when this request finished.
	EndTime time.Time

	// Opaque cursor into the result stream.
	Offset []byte

	// The time required to process the request.
	Latency     time.Duration
	MCycles     int64
	Method      string
	Resource    string
	HTTPVersion string
	Status      int32

	// The size of the request sent back to the client, in bytes.
	ResponseSize int64
	Referrer     string
	UserAgent    string
	URLMapEntry  string
	Combined     string
	Host         string

	// The estimated cost of this request, in dollars.
	Cost              float64
	TaskQueueName     string
	TaskName          string
	WasLoadingRequest bool
	PendingTime       time.Duration
	Finished          bool
	AppLogs           []AppLog

	// Mostly-unique identifier for the instance that handled the request if available.
	InstanceID string
}

// Result represents the result of a query.
type Result struct {
	logs        []*Record
	context     context.Context
	request     *pb.LogReadRequest
	resultsSeen bool
	err         error
}

// Next returns the next log record,
func (qr *Result) Next() (*Record, error) {
	if qr.err != nil {
		return nil, qr.err
	}
	if len(qr.logs) > 0 {
		lr := qr.logs[0]
		qr.logs = qr.logs[1:]
		return lr, nil
	}

	if qr.request.Offset == nil && qr.resultsSeen {
		return nil, Done
	}

	if err := qr.run(); err != nil {
		// Errors here may be retried, so don't store the error.
		return nil, err
	}

	return qr.Next()
}

// Done is returned when a query iteration has completed.
var Done = errors.New("log: query has no more results")

// protoToAppLogs takes as input an array of pointers to LogLines, the internal
// Protocol Buffer representation of a single application-level log,
// and converts it to an array of AppLogs, the external representation
// of an application-level log.
func protoToAppLogs(logLines []*pb.LogLine) []AppLog {
	appLogs := make([]AppLog, len(logLines))

	for i, line := range logLines {
		appLogs[i] = AppLog{
			Time:    time.Unix(0, *line.Time*1e3),
			Level:   int(*line.Level),
			Message: *line.LogMessage,
		}
	}

	return appLogs
}

// protoToRecord converts a RequestLog, the internal Protocol Buffer
// representation of a single request-level log, to a Record, its
// corresponding external representation.
func protoToRecord(rl *pb.RequestLog) *Record {
	offset, err := proto.Marshal(rl.Offset)
	if err != nil {
		offset = nil
	}
	return &Record{
		AppID:             *rl.AppId,
		ModuleID:          rl.GetModuleId(),
		VersionID:         *rl.VersionId,
		RequestID:         rl.RequestId,
		Offset:            offset,
		IP:                *rl.Ip,
		Nickname:          rl.GetNickname(),
		AppEngineRelease:  string(rl.GetAppEngineRelease()),
		StartTime:         time.Unix(0, *rl.StartTime*1e3),
		EndTime:           time.Unix(0, *rl.EndTime*1e3),
		Latency:           time.Duration(*rl.Latency) * time.Microsecond,
		MCycles:           *rl.Mcycles,
		Method:            *rl.Method,
		Resource:          *rl.Resource,
		HTTPVersion:       *rl.HttpVersion,
		Status:            *rl.Status,
		ResponseSize:      *rl.ResponseSize,
		Referrer:          rl.GetReferrer(),
		UserAgent:         rl.GetUserAgent(),
		URLMapEntry:       *rl.UrlMapEntry,
		Combined:          *rl.Combined,
		Host:              rl.GetHost(),
		Cost:              rl.GetCost(),
		TaskQueueName:     rl.GetTaskQueueName(),
		TaskName:          rl.GetTaskName(),
		WasLoadingRequest: rl.GetWasLoadingRequest(),
		PendingTime:       time.Duration(rl.GetPendingTime()) * time.Microsecond,
		Finished:          rl.GetFinished(),
		AppLogs:           protoToAppLogs(rl.Line),
		InstanceID:        string(rl.GetCloneKey()),
	}
}

// Run starts a query for log records, which contain request and application
// level log information.
func (params *Query) Run(c context.Context) *Result {
	req, err := makeRequest(params, internal.FullyQualifiedAppID(c), appengine.VersionID(c))
	return &Result{
		context: c,
		request: req,
		err:     err,
	}
}

func makeRequest(params *Query, appID, versionID string) (*pb.LogReadRequest, error) {
	req := &pb.LogReadRequest{}
	req.AppId = &appID
	if !params.StartTime.IsZero() {
		req.StartTime = proto.Int64(params.StartTime.UnixNano() / 1e3)
	}
	if !params.EndTime.IsZero() {
		req.EndTime = proto.Int64(params.EndTime.UnixNano() / 1e3)
	}
	if len(params.Offset) > 0 {
		var offset pb.LogOffset
		if err := proto.Unmarshal(params.Offset, &offset); err != nil {
			return nil, fmt.Errorf("bad Offset: %v", err)
		}
		req.Offset = &offset
	}
	if params.Incomplete {
		req.IncludeIncomplete = &params.Incomplete
	}
	if params.AppLogs {
		req.IncludeAppLogs = &params.AppLogs
	}
	if params.ApplyMinLevel {
		req.MinimumLogLevel = proto.Int32(int32(params.MinLevel))
	}
	if params.Versions == nil {
		// If no versions were specified, default to the default module at
		// the major version being used by this module.
		if i := strings.Index(versionID, "."); i >= 0 {
			versionID = versionID[:i]
		}
		req.VersionId = []string{versionID}
	} else {
		req.ModuleVersion = make([]*pb.LogModuleVersion, 0, len(params.Versions))
		for _, v := range params.Versions {
			var m *string
			if i := strings.Index(v, ":"); i >= 0 {
				m, v = proto.String(v[:i]), v[i+1:]
			}
			req.ModuleVersion = append(req.ModuleVersion, &pb.LogModuleVersion{
				ModuleId:  m,
				VersionId: proto.String(v),
			})
		}
	}
	if params.RequestIDs != nil {
		ids := make([][]byte, len(params.RequestIDs))
		for i, v := range params.RequestIDs {
			ids[i] = []byte(v)
		}
		req.RequestId = ids
	}

	return req, nil
}

// run takes the query Result produced by a call to Run and updates it with
// more Records. The updated Result contains a new set of logs as well as an
// offset to where more logs can be found. We also convert the items in the
// response from their internal representations to external versions of the
// same structs.
func (r *Result) run() error {
	res := &pb.LogReadResponse{}
	if err := internal.Call(r.context, "logservice", "Read", r.request, res); err != nil {
		return err
	}

	r.logs = make([]*Record, len(res.Log))
	r.request.Offset = res.Offset
	r.resultsSeen = true

	for i, log := range res.Log {
		r.logs[i] = protoToRecord(log)
	}

	return nil
}

func init() {
	internal.RegisterErrorCodeMap("logservice", pb.LogServiceError_ErrorCode_name)
}
