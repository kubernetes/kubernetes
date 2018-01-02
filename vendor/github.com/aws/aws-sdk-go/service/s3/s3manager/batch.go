package s3manager

import (
	"bytes"
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

const (
	// DefaultBatchSize is the batch size we initialize when constructing a batch delete client.
	// This value is used when calling DeleteObjects. This represents how many objects to delete
	// per DeleteObjects call.
	DefaultBatchSize = 100
)

// BatchError will contain the key and bucket of the object that failed to
// either upload or download.
type BatchError struct {
	Errors  Errors
	code    string
	message string
}

// Errors is a typed alias for a slice of errors to satisfy the error
// interface.
type Errors []Error

func (errs Errors) Error() string {
	buf := bytes.NewBuffer(nil)
	for i, err := range errs {
		buf.WriteString(err.Error())
		if i+1 < len(errs) {
			buf.WriteString("\n")
		}
	}
	return buf.String()
}

// Error will contain the original error, bucket, and key of the operation that failed
// during batch operations.
type Error struct {
	OrigErr error
	Bucket  *string
	Key     *string
}

func newError(err error, bucket, key *string) Error {
	return Error{
		err,
		bucket,
		key,
	}
}

func (err *Error) Error() string {
	return fmt.Sprintf("failed to upload %q to %q:\n%s", err.Key, err.Bucket, err.OrigErr.Error())
}

// NewBatchError will return a BatchError that satisfies the awserr.Error interface.
func NewBatchError(code, message string, err []Error) awserr.Error {
	return &BatchError{
		Errors:  err,
		code:    code,
		message: message,
	}
}

// Code will return the code associated with the batch error.
func (err *BatchError) Code() string {
	return err.code
}

// Message will return the message associated with the batch error.
func (err *BatchError) Message() string {
	return err.message
}

func (err *BatchError) Error() string {
	return awserr.SprintError(err.Code(), err.Message(), "", err.Errors)
}

// OrigErr will return the original error. Which, in this case, will always be nil
// for batched operations.
func (err *BatchError) OrigErr() error {
	return err.Errors
}

// BatchDeleteIterator is an interface that uses the scanner pattern to
// iterate through what needs to be deleted.
type BatchDeleteIterator interface {
	Next() bool
	Err() error
	DeleteObject() BatchDeleteObject
}

// DeleteListIterator is an alternative iterator for the BatchDelete client. This will
// iterate through a list of objects and delete the objects.
//
// Example:
//	iter := &s3manager.DeleteListIterator{
//		Client: svc,
//		Input: &s3.ListObjectsInput{
//			Bucket:  aws.String("bucket"),
//			MaxKeys: aws.Int64(5),
//		},
//		Paginator: request.Pagination{
//			NewRequest: func() (*request.Request, error) {
//				var inCpy *ListObjectsInput
//				if input != nil {
//					tmp := *input
//					inCpy = &tmp
//				}
//				req, _ := c.ListObjectsRequest(inCpy)
//				return req, nil
//			},
//		},
//	}
//
//	batcher := s3manager.NewBatchDeleteWithClient(svc)
//	if err := batcher.Delete(aws.BackgroundContext(), iter); err != nil {
//		return err
//	}
type DeleteListIterator struct {
	Bucket    *string
	Paginator request.Pagination
	objects   []*s3.Object
}

// NewDeleteListIterator will return a new DeleteListIterator.
func NewDeleteListIterator(svc s3iface.S3API, input *s3.ListObjectsInput, opts ...func(*DeleteListIterator)) BatchDeleteIterator {
	iter := &DeleteListIterator{
		Bucket: input.Bucket,
		Paginator: request.Pagination{
			NewRequest: func() (*request.Request, error) {
				var inCpy *s3.ListObjectsInput
				if input != nil {
					tmp := *input
					inCpy = &tmp
				}
				req, _ := svc.ListObjectsRequest(inCpy)
				return req, nil
			},
		},
	}

	for _, opt := range opts {
		opt(iter)
	}
	return iter
}

// Next will use the S3API client to iterate through a list of objects.
func (iter *DeleteListIterator) Next() bool {
	if len(iter.objects) > 0 {
		iter.objects = iter.objects[1:]
	}

	if len(iter.objects) == 0 && iter.Paginator.Next() {
		iter.objects = iter.Paginator.Page().(*s3.ListObjectsOutput).Contents
	}

	return len(iter.objects) > 0
}

// Err will return the last known error from Next.
func (iter *DeleteListIterator) Err() error {
	return iter.Paginator.Err()
}

// DeleteObject will return the current object to be deleted.
func (iter *DeleteListIterator) DeleteObject() BatchDeleteObject {
	return BatchDeleteObject{
		Object: &s3.DeleteObjectInput{
			Bucket: iter.Bucket,
			Key:    iter.objects[0].Key,
		},
	}
}

// BatchDelete will use the s3 package's service client to perform a batch
// delete.
type BatchDelete struct {
	Client    s3iface.S3API
	BatchSize int
}

// NewBatchDeleteWithClient will return a new delete client that can delete a batched amount of
// objects.
//
// Example:
//	batcher := s3manager.NewBatchDeleteWithClient(client, size)
//
//	objects := []BatchDeleteObject{
//		{
//			Object:	&s3.DeleteObjectInput {
//				Key: aws.String("key"),
//				Bucket: aws.String("bucket"),
//			},
//		},
//	}
//
//	if err := batcher.Delete(&s3manager.DeleteObjectsIterator{
//		Objects: objects,
//	}); err != nil {
//		return err
//	}
func NewBatchDeleteWithClient(client s3iface.S3API, options ...func(*BatchDelete)) *BatchDelete {
	svc := &BatchDelete{
		Client:    client,
		BatchSize: DefaultBatchSize,
	}

	for _, opt := range options {
		opt(svc)
	}

	return svc
}

// NewBatchDelete will return a new delete client that can delete a batched amount of
// objects.
//
// Example:
//	batcher := s3manager.NewBatchDelete(sess, size)
//
//	objects := []BatchDeleteObject{
//		{
//			Object:	&s3.DeleteObjectInput {
//				Key: aws.String("key"),
//				Bucket: aws.String("bucket"),
//			},
//		},
//	}
//
//	if err := batcher.Delete(&s3manager.DeleteObjectsIterator{
//		Objects: objects,
//	}); err != nil {
//		return err
//	}
func NewBatchDelete(c client.ConfigProvider, options ...func(*BatchDelete)) *BatchDelete {
	client := s3.New(c)
	return NewBatchDeleteWithClient(client, options...)
}

// BatchDeleteObject is a wrapper object for calling the batch delete operation.
type BatchDeleteObject struct {
	Object *s3.DeleteObjectInput
	// After will run after each iteration during the batch process. This function will
	// be executed whether or not the request was successful.
	After func() error
}

// DeleteObjectsIterator is an interface that uses the scanner pattern to iterate
// through a series of objects to be deleted.
type DeleteObjectsIterator struct {
	Objects []BatchDeleteObject
	index   int
	inc     bool
}

// Next will increment the default iterator's index and and ensure that there
// is another object to iterator to.
func (iter *DeleteObjectsIterator) Next() bool {
	if iter.inc {
		iter.index++
	} else {
		iter.inc = true
	}
	return iter.index < len(iter.Objects)
}

// Err will return an error. Since this is just used to satisfy the BatchDeleteIterator interface
// this will only return nil.
func (iter *DeleteObjectsIterator) Err() error {
	return nil
}

// DeleteObject will return the BatchDeleteObject at the current batched index.
func (iter *DeleteObjectsIterator) DeleteObject() BatchDeleteObject {
	object := iter.Objects[iter.index]
	return object
}

// Delete will use the iterator to queue up objects that need to be deleted.
// Once the batch size is met, this will call the deleteBatch function.
func (d *BatchDelete) Delete(ctx aws.Context, iter BatchDeleteIterator) error {
	var errs []Error
	objects := []BatchDeleteObject{}
	var input *s3.DeleteObjectsInput

	for iter.Next() {
		o := iter.DeleteObject()

		if input == nil {
			input = initDeleteObjectsInput(o.Object)
		}

		parity := hasParity(input, o)
		if parity {
			input.Delete.Objects = append(input.Delete.Objects, &s3.ObjectIdentifier{
				Key:       o.Object.Key,
				VersionId: o.Object.VersionId,
			})
			objects = append(objects, o)
		}

		if len(input.Delete.Objects) == d.BatchSize || !parity {
			if err := deleteBatch(d, input, objects); err != nil {
				errs = append(errs, err...)
			}

			objects = objects[:0]
			input = nil

			if !parity {
				objects = append(objects, o)
				input = initDeleteObjectsInput(o.Object)
				input.Delete.Objects = append(input.Delete.Objects, &s3.ObjectIdentifier{
					Key:       o.Object.Key,
					VersionId: o.Object.VersionId,
				})
			}
		}
	}

	if input != nil && len(input.Delete.Objects) > 0 {
		if err := deleteBatch(d, input, objects); err != nil {
			errs = append(errs, err...)
		}
	}

	if len(errs) > 0 {
		return NewBatchError("BatchedDeleteIncomplete", "some objects have failed to be deleted.", errs)
	}
	return nil
}

func initDeleteObjectsInput(o *s3.DeleteObjectInput) *s3.DeleteObjectsInput {
	return &s3.DeleteObjectsInput{
		Bucket:       o.Bucket,
		MFA:          o.MFA,
		RequestPayer: o.RequestPayer,
		Delete:       &s3.Delete{},
	}
}

// deleteBatch will delete a batch of items in the objects parameters.
func deleteBatch(d *BatchDelete, input *s3.DeleteObjectsInput, objects []BatchDeleteObject) []Error {
	errs := []Error{}

	if result, err := d.Client.DeleteObjects(input); err != nil {
		for i := 0; i < len(input.Delete.Objects); i++ {
			errs = append(errs, newError(err, input.Bucket, input.Delete.Objects[i].Key))
		}
	} else if len(result.Errors) > 0 {
		for i := 0; i < len(result.Errors); i++ {
			errs = append(errs, newError(err, input.Bucket, result.Errors[i].Key))
		}
	}
	for _, object := range objects {
		if object.After == nil {
			continue
		}
		if err := object.After(); err != nil {
			errs = append(errs, newError(err, object.Object.Bucket, object.Object.Key))
		}
	}

	return errs
}

func hasParity(o1 *s3.DeleteObjectsInput, o2 BatchDeleteObject) bool {
	if o1.Bucket != nil && o2.Object.Bucket != nil {
		if *o1.Bucket != *o2.Object.Bucket {
			return false
		}
	} else if o1.Bucket != o2.Object.Bucket {
		return false
	}

	if o1.MFA != nil && o2.Object.MFA != nil {
		if *o1.MFA != *o2.Object.MFA {
			return false
		}
	} else if o1.MFA != o2.Object.MFA {
		return false
	}

	if o1.RequestPayer != nil && o2.Object.RequestPayer != nil {
		if *o1.RequestPayer != *o2.Object.RequestPayer {
			return false
		}
	} else if o1.RequestPayer != o2.Object.RequestPayer {
		return false
	}

	return true
}

// BatchDownloadIterator is an interface that uses the scanner pattern to iterate
// through a series of objects to be downloaded.
type BatchDownloadIterator interface {
	Next() bool
	Err() error
	DownloadObject() BatchDownloadObject
}

// BatchDownloadObject contains all necessary information to run a batch operation once.
type BatchDownloadObject struct {
	Object *s3.GetObjectInput
	Writer io.WriterAt
	// After will run after each iteration during the batch process. This function will
	// be executed whether or not the request was successful.
	After func() error
}

// DownloadObjectsIterator implements the BatchDownloadIterator interface and allows for batched
// download of objects.
type DownloadObjectsIterator struct {
	Objects []BatchDownloadObject
	index   int
	inc     bool
}

// Next will increment the default iterator's index and and ensure that there
// is another object to iterator to.
func (batcher *DownloadObjectsIterator) Next() bool {
	if batcher.inc {
		batcher.index++
	} else {
		batcher.inc = true
	}
	return batcher.index < len(batcher.Objects)
}

// DownloadObject will return the BatchDownloadObject at the current batched index.
func (batcher *DownloadObjectsIterator) DownloadObject() BatchDownloadObject {
	object := batcher.Objects[batcher.index]
	return object
}

// Err will return an error. Since this is just used to satisfy the BatchDeleteIterator interface
// this will only return nil.
func (batcher *DownloadObjectsIterator) Err() error {
	return nil
}

// BatchUploadIterator is an interface that uses the scanner pattern to
// iterate through what needs to be uploaded.
type BatchUploadIterator interface {
	Next() bool
	Err() error
	UploadObject() BatchUploadObject
}

// UploadObjectsIterator implements the BatchUploadIterator interface and allows for batched
// upload of objects.
type UploadObjectsIterator struct {
	Objects []BatchUploadObject
	index   int
	inc     bool
}

// Next will increment the default iterator's index and and ensure that there
// is another object to iterator to.
func (batcher *UploadObjectsIterator) Next() bool {
	if batcher.inc {
		batcher.index++
	} else {
		batcher.inc = true
	}
	return batcher.index < len(batcher.Objects)
}

// Err will return an error. Since this is just used to satisfy the BatchUploadIterator interface
// this will only return nil.
func (batcher *UploadObjectsIterator) Err() error {
	return nil
}

// UploadObject will return the BatchUploadObject at the current batched index.
func (batcher *UploadObjectsIterator) UploadObject() BatchUploadObject {
	object := batcher.Objects[batcher.index]
	return object
}

// BatchUploadObject contains all necessary information to run a batch operation once.
type BatchUploadObject struct {
	Object *UploadInput
	// After will run after each iteration during the batch process. This function will
	// be executed whether or not the request was successful.
	After func() error
}
