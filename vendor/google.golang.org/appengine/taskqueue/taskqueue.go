// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package taskqueue provides a client for App Engine's taskqueue service.
Using this service, applications may perform work outside a user's request.

A Task may be constructed manually; alternatively, since the most common
taskqueue operation is to add a single POST task, NewPOSTTask makes it easy.

	t := taskqueue.NewPOSTTask("/worker", url.Values{
		"key": {key},
	})
	taskqueue.Add(c, t, "") // add t to the default queue
*/
package taskqueue // import "google.golang.org/appengine/taskqueue"

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	dspb "google.golang.org/appengine/internal/datastore"
	pb "google.golang.org/appengine/internal/taskqueue"
)

var (
	// ErrTaskAlreadyAdded is the error returned by Add and AddMulti when a task has already been added with a particular name.
	ErrTaskAlreadyAdded = errors.New("taskqueue: task has already been added")
)

// RetryOptions let you control whether to retry a task and the backoff intervals between tries.
type RetryOptions struct {
	// Number of tries/leases after which the task fails permanently and is deleted.
	// If AgeLimit is also set, both limits must be exceeded for the task to fail permanently.
	RetryLimit int32

	// Maximum time allowed since the task's first try before the task fails permanently and is deleted (only for push tasks).
	// If RetryLimit is also set, both limits must be exceeded for the task to fail permanently.
	AgeLimit time.Duration

	// Minimum time between successive tries (only for push tasks).
	MinBackoff time.Duration

	// Maximum time between successive tries (only for push tasks).
	MaxBackoff time.Duration

	// Maximum number of times to double the interval between successive tries before the intervals increase linearly (only for push tasks).
	MaxDoublings int32

	// If MaxDoublings is zero, set ApplyZeroMaxDoublings to true to override the default non-zero value.
	// Otherwise a zero MaxDoublings is ignored and the default is used.
	ApplyZeroMaxDoublings bool
}

// toRetryParameter converts RetryOptions to pb.TaskQueueRetryParameters.
func (opt *RetryOptions) toRetryParameters() *pb.TaskQueueRetryParameters {
	params := &pb.TaskQueueRetryParameters{}
	if opt.RetryLimit > 0 {
		params.RetryLimit = proto.Int32(opt.RetryLimit)
	}
	if opt.AgeLimit > 0 {
		params.AgeLimitSec = proto.Int64(int64(opt.AgeLimit.Seconds()))
	}
	if opt.MinBackoff > 0 {
		params.MinBackoffSec = proto.Float64(opt.MinBackoff.Seconds())
	}
	if opt.MaxBackoff > 0 {
		params.MaxBackoffSec = proto.Float64(opt.MaxBackoff.Seconds())
	}
	if opt.MaxDoublings > 0 || (opt.MaxDoublings == 0 && opt.ApplyZeroMaxDoublings) {
		params.MaxDoublings = proto.Int32(opt.MaxDoublings)
	}
	return params
}

// A Task represents a task to be executed.
type Task struct {
	// Path is the worker URL for the task.
	// If unset, it will default to /_ah/queue/<queue_name>.
	Path string

	// Payload is the data for the task.
	// This will be delivered as the HTTP request body.
	// It is only used when Method is POST, PUT or PULL.
	// url.Values' Encode method may be used to generate this for POST requests.
	Payload []byte

	// Additional HTTP headers to pass at the task's execution time.
	// To schedule the task to be run with an alternate app version
	// or backend, set the "Host" header.
	Header http.Header

	// Method is the HTTP method for the task ("GET", "POST", etc.),
	// or "PULL" if this is task is destined for a pull-based queue.
	// If empty, this defaults to "POST".
	Method string

	// A name for the task.
	// If empty, a name will be chosen.
	Name string

	// Delay specifies the duration the task queue service must wait
	// before executing the task.
	// Either Delay or ETA may be set, but not both.
	Delay time.Duration

	// ETA specifies the earliest time a task may be executed (push queues)
	// or leased (pull queues).
	// Either Delay or ETA may be set, but not both.
	ETA time.Time

	// The number of times the task has been dispatched or leased.
	RetryCount int32

	// Tag for the task. Only used when Method is PULL.
	Tag string

	// Retry options for this task. May be nil.
	RetryOptions *RetryOptions
}

func (t *Task) method() string {
	if t.Method == "" {
		return "POST"
	}
	return t.Method
}

// NewPOSTTask creates a Task that will POST to a path with the given form data.
func NewPOSTTask(path string, params url.Values) *Task {
	h := make(http.Header)
	h.Set("Content-Type", "application/x-www-form-urlencoded")
	return &Task{
		Path:    path,
		Payload: []byte(params.Encode()),
		Header:  h,
		Method:  "POST",
	}
}

var (
	currentNamespace = http.CanonicalHeaderKey("X-AppEngine-Current-Namespace")
	defaultNamespace = http.CanonicalHeaderKey("X-AppEngine-Default-Namespace")
)

func getDefaultNamespace(ctx context.Context) string {
	return internal.IncomingHeaders(ctx).Get(defaultNamespace)
}

func newAddReq(c context.Context, task *Task, queueName string) (*pb.TaskQueueAddRequest, error) {
	if queueName == "" {
		queueName = "default"
	}
	path := task.Path
	if path == "" {
		path = "/_ah/queue/" + queueName
	}
	eta := task.ETA
	if eta.IsZero() {
		eta = time.Now().Add(task.Delay)
	} else if task.Delay != 0 {
		panic("taskqueue: both Delay and ETA are set")
	}
	req := &pb.TaskQueueAddRequest{
		QueueName: []byte(queueName),
		TaskName:  []byte(task.Name),
		EtaUsec:   proto.Int64(eta.UnixNano() / 1e3),
	}
	method := task.method()
	if method == "PULL" {
		// Pull-based task
		req.Body = task.Payload
		req.Mode = pb.TaskQueueMode_PULL.Enum()
		if task.Tag != "" {
			req.Tag = []byte(task.Tag)
		}
	} else {
		// HTTP-based task
		if v, ok := pb.TaskQueueAddRequest_RequestMethod_value[method]; ok {
			req.Method = pb.TaskQueueAddRequest_RequestMethod(v).Enum()
		} else {
			return nil, fmt.Errorf("taskqueue: bad method %q", method)
		}
		req.Url = []byte(path)
		for k, vs := range task.Header {
			for _, v := range vs {
				req.Header = append(req.Header, &pb.TaskQueueAddRequest_Header{
					Key:   []byte(k),
					Value: []byte(v),
				})
			}
		}
		if method == "POST" || method == "PUT" {
			req.Body = task.Payload
		}

		// Namespace headers.
		if _, ok := task.Header[currentNamespace]; !ok {
			// Fetch the current namespace of this request.
			ns := internal.NamespaceFromContext(c)
			req.Header = append(req.Header, &pb.TaskQueueAddRequest_Header{
				Key:   []byte(currentNamespace),
				Value: []byte(ns),
			})
		}
		if _, ok := task.Header[defaultNamespace]; !ok {
			// Fetch the X-AppEngine-Default-Namespace header of this request.
			if ns := getDefaultNamespace(c); ns != "" {
				req.Header = append(req.Header, &pb.TaskQueueAddRequest_Header{
					Key:   []byte(defaultNamespace),
					Value: []byte(ns),
				})
			}
		}
	}

	if task.RetryOptions != nil {
		req.RetryParameters = task.RetryOptions.toRetryParameters()
	}

	return req, nil
}

var alreadyAddedErrors = map[pb.TaskQueueServiceError_ErrorCode]bool{
	pb.TaskQueueServiceError_TASK_ALREADY_EXISTS: true,
	pb.TaskQueueServiceError_TOMBSTONED_TASK:     true,
}

// Add adds the task to a named queue.
// An empty queue name means that the default queue will be used.
// Add returns an equivalent Task with defaults filled in, including setting
// the task's Name field to the chosen name if the original was empty.
func Add(c context.Context, task *Task, queueName string) (*Task, error) {
	req, err := newAddReq(c, task, queueName)
	if err != nil {
		return nil, err
	}
	res := &pb.TaskQueueAddResponse{}
	if err := internal.Call(c, "taskqueue", "Add", req, res); err != nil {
		apiErr, ok := err.(*internal.APIError)
		if ok && alreadyAddedErrors[pb.TaskQueueServiceError_ErrorCode(apiErr.Code)] {
			return nil, ErrTaskAlreadyAdded
		}
		return nil, err
	}
	resultTask := *task
	resultTask.Method = task.method()
	if task.Name == "" {
		resultTask.Name = string(res.ChosenTaskName)
	}
	return &resultTask, nil
}

// AddMulti adds multiple tasks to a named queue.
// An empty queue name means that the default queue will be used.
// AddMulti returns a slice of equivalent tasks with defaults filled in, including setting
// each task's Name field to the chosen name if the original was empty.
// If a given task is badly formed or could not be added, an appengine.MultiError is returned.
func AddMulti(c context.Context, tasks []*Task, queueName string) ([]*Task, error) {
	req := &pb.TaskQueueBulkAddRequest{
		AddRequest: make([]*pb.TaskQueueAddRequest, len(tasks)),
	}
	me, any := make(appengine.MultiError, len(tasks)), false
	for i, t := range tasks {
		req.AddRequest[i], me[i] = newAddReq(c, t, queueName)
		any = any || me[i] != nil
	}
	if any {
		return nil, me
	}
	res := &pb.TaskQueueBulkAddResponse{}
	if err := internal.Call(c, "taskqueue", "BulkAdd", req, res); err != nil {
		return nil, err
	}
	if len(res.Taskresult) != len(tasks) {
		return nil, errors.New("taskqueue: server error")
	}
	tasksOut := make([]*Task, len(tasks))
	for i, tr := range res.Taskresult {
		tasksOut[i] = new(Task)
		*tasksOut[i] = *tasks[i]
		tasksOut[i].Method = tasksOut[i].method()
		if tasksOut[i].Name == "" {
			tasksOut[i].Name = string(tr.ChosenTaskName)
		}
		if *tr.Result != pb.TaskQueueServiceError_OK {
			if alreadyAddedErrors[*tr.Result] {
				me[i] = ErrTaskAlreadyAdded
			} else {
				me[i] = &internal.APIError{
					Service: "taskqueue",
					Code:    int32(*tr.Result),
				}
			}
			any = true
		}
	}
	if any {
		return tasksOut, me
	}
	return tasksOut, nil
}

// Delete deletes a task from a named queue.
func Delete(c context.Context, task *Task, queueName string) error {
	err := DeleteMulti(c, []*Task{task}, queueName)
	if me, ok := err.(appengine.MultiError); ok {
		return me[0]
	}
	return err
}

// DeleteMulti deletes multiple tasks from a named queue.
// If a given task could not be deleted, an appengine.MultiError is returned.
func DeleteMulti(c context.Context, tasks []*Task, queueName string) error {
	taskNames := make([][]byte, len(tasks))
	for i, t := range tasks {
		taskNames[i] = []byte(t.Name)
	}
	if queueName == "" {
		queueName = "default"
	}
	req := &pb.TaskQueueDeleteRequest{
		QueueName: []byte(queueName),
		TaskName:  taskNames,
	}
	res := &pb.TaskQueueDeleteResponse{}
	if err := internal.Call(c, "taskqueue", "Delete", req, res); err != nil {
		return err
	}
	if a, b := len(req.TaskName), len(res.Result); a != b {
		return fmt.Errorf("taskqueue: internal error: requested deletion of %d tasks, got %d results", a, b)
	}
	me, any := make(appengine.MultiError, len(res.Result)), false
	for i, ec := range res.Result {
		if ec != pb.TaskQueueServiceError_OK {
			me[i] = &internal.APIError{
				Service: "taskqueue",
				Code:    int32(ec),
			}
			any = true
		}
	}
	if any {
		return me
	}
	return nil
}

func lease(c context.Context, maxTasks int, queueName string, leaseTime int, groupByTag bool, tag []byte) ([]*Task, error) {
	if queueName == "" {
		queueName = "default"
	}
	req := &pb.TaskQueueQueryAndOwnTasksRequest{
		QueueName:    []byte(queueName),
		LeaseSeconds: proto.Float64(float64(leaseTime)),
		MaxTasks:     proto.Int64(int64(maxTasks)),
		GroupByTag:   proto.Bool(groupByTag),
		Tag:          tag,
	}
	res := &pb.TaskQueueQueryAndOwnTasksResponse{}
	if err := internal.Call(c, "taskqueue", "QueryAndOwnTasks", req, res); err != nil {
		return nil, err
	}
	tasks := make([]*Task, len(res.Task))
	for i, t := range res.Task {
		tasks[i] = &Task{
			Payload:    t.Body,
			Name:       string(t.TaskName),
			Method:     "PULL",
			ETA:        time.Unix(0, *t.EtaUsec*1e3),
			RetryCount: *t.RetryCount,
			Tag:        string(t.Tag),
		}
	}
	return tasks, nil
}

// Lease leases tasks from a queue.
// leaseTime is in seconds.
// The number of tasks fetched will be at most maxTasks.
func Lease(c context.Context, maxTasks int, queueName string, leaseTime int) ([]*Task, error) {
	return lease(c, maxTasks, queueName, leaseTime, false, nil)
}

// LeaseByTag leases tasks from a queue, grouped by tag.
// If tag is empty, then the returned tasks are grouped by the tag of the task with earliest ETA.
// leaseTime is in seconds.
// The number of tasks fetched will be at most maxTasks.
func LeaseByTag(c context.Context, maxTasks int, queueName string, leaseTime int, tag string) ([]*Task, error) {
	return lease(c, maxTasks, queueName, leaseTime, true, []byte(tag))
}

// Purge removes all tasks from a queue.
func Purge(c context.Context, queueName string) error {
	if queueName == "" {
		queueName = "default"
	}
	req := &pb.TaskQueuePurgeQueueRequest{
		QueueName: []byte(queueName),
	}
	res := &pb.TaskQueuePurgeQueueResponse{}
	return internal.Call(c, "taskqueue", "PurgeQueue", req, res)
}

// ModifyLease modifies the lease of a task.
// Used to request more processing time, or to abandon processing.
// leaseTime is in seconds and must not be negative.
func ModifyLease(c context.Context, task *Task, queueName string, leaseTime int) error {
	if queueName == "" {
		queueName = "default"
	}
	req := &pb.TaskQueueModifyTaskLeaseRequest{
		QueueName:    []byte(queueName),
		TaskName:     []byte(task.Name),
		EtaUsec:      proto.Int64(task.ETA.UnixNano() / 1e3), // Used to verify ownership.
		LeaseSeconds: proto.Float64(float64(leaseTime)),
	}
	res := &pb.TaskQueueModifyTaskLeaseResponse{}
	if err := internal.Call(c, "taskqueue", "ModifyTaskLease", req, res); err != nil {
		return err
	}
	task.ETA = time.Unix(0, *res.UpdatedEtaUsec*1e3)
	return nil
}

// QueueStatistics represents statistics about a single task queue.
type QueueStatistics struct {
	Tasks     int       // may be an approximation
	OldestETA time.Time // zero if there are no pending tasks

	Executed1Minute int     // tasks executed in the last minute
	InFlight        int     // tasks executing now
	EnforcedRate    float64 // requests per second
}

// QueueStats retrieves statistics about queues.
func QueueStats(c context.Context, queueNames []string) ([]QueueStatistics, error) {
	req := &pb.TaskQueueFetchQueueStatsRequest{
		QueueName: make([][]byte, len(queueNames)),
	}
	for i, q := range queueNames {
		if q == "" {
			q = "default"
		}
		req.QueueName[i] = []byte(q)
	}
	res := &pb.TaskQueueFetchQueueStatsResponse{}
	if err := internal.Call(c, "taskqueue", "FetchQueueStats", req, res); err != nil {
		return nil, err
	}
	qs := make([]QueueStatistics, len(res.Queuestats))
	for i, qsg := range res.Queuestats {
		qs[i] = QueueStatistics{
			Tasks: int(*qsg.NumTasks),
		}
		if eta := *qsg.OldestEtaUsec; eta > -1 {
			qs[i].OldestETA = time.Unix(0, eta*1e3)
		}
		if si := qsg.ScannerInfo; si != nil {
			qs[i].Executed1Minute = int(*si.ExecutedLastMinute)
			qs[i].InFlight = int(si.GetRequestsInFlight())
			qs[i].EnforcedRate = si.GetEnforcedRate()
		}
	}
	return qs, nil
}

func setTransaction(x *pb.TaskQueueAddRequest, t *dspb.Transaction) {
	x.Transaction = t
}

func init() {
	internal.RegisterErrorCodeMap("taskqueue", pb.TaskQueueServiceError_ErrorCode_name)

	// Datastore error codes are shifted by DATASTORE_ERROR when presented through taskqueue.
	dsCode := int32(pb.TaskQueueServiceError_DATASTORE_ERROR) + int32(dspb.Error_TIMEOUT)
	internal.RegisterTimeoutErrorCode("taskqueue", dsCode)

	// Transaction registration.
	internal.RegisterTransactionSetter(setTransaction)
	internal.RegisterTransactionSetter(func(x *pb.TaskQueueBulkAddRequest, t *dspb.Transaction) {
		for _, req := range x.AddRequest {
			setTransaction(req, t)
		}
	})
}
