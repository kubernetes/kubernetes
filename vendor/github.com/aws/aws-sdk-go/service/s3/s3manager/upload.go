package s3manager

import (
	"bytes"
	"fmt"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

// MaxUploadParts is the maximum allowed number of parts in a multi-part upload
// on Amazon S3.
const MaxUploadParts = 10000

// MinUploadPartSize is the minimum allowed part size when uploading a part to
// Amazon S3.
const MinUploadPartSize int64 = 1024 * 1024 * 5

// DefaultUploadPartSize is the default part size to buffer chunks of a
// payload into.
const DefaultUploadPartSize = MinUploadPartSize

// DefaultUploadConcurrency is the default number of goroutines to spin up when
// using Upload().
const DefaultUploadConcurrency = 5

// A MultiUploadFailure wraps a failed S3 multipart upload. An error returned
// will satisfy this interface when a multi part upload failed to upload all
// chucks to S3. In the case of a failure the UploadID is needed to operate on
// the chunks, if any, which were uploaded.
//
// Example:
//
//     u := s3manager.NewUploader(opts)
//     output, err := u.upload(input)
//     if err != nil {
//         if multierr, ok := err.(MultiUploadFailure); ok {
//             // Process error and its associated uploadID
//             fmt.Println("Error:", multierr.Code(), multierr.Message(), multierr.UploadID())
//         } else {
//             // Process error generically
//             fmt.Println("Error:", err.Error())
//         }
//     }
//
type MultiUploadFailure interface {
	awserr.Error

	// Returns the upload id for the S3 multipart upload that failed.
	UploadID() string
}

// So that the Error interface type can be included as an anonymous field
// in the multiUploadError struct and not conflict with the error.Error() method.
type awsError awserr.Error

// A multiUploadError wraps the upload ID of a failed s3 multipart upload.
// Composed of BaseError for code, message, and original error
//
// Should be used for an error that occurred failing a S3 multipart upload,
// and a upload ID is available. If an uploadID is not available a more relevant
type multiUploadError struct {
	awsError

	// ID for multipart upload which failed.
	uploadID string
}

// Error returns the string representation of the error.
//
// See apierr.BaseError ErrorWithExtra for output format
//
// Satisfies the error interface.
func (m multiUploadError) Error() string {
	extra := fmt.Sprintf("upload id: %s", m.uploadID)
	return awserr.SprintError(m.Code(), m.Message(), extra, m.OrigErr())
}

// String returns the string representation of the error.
// Alias for Error to satisfy the stringer interface.
func (m multiUploadError) String() string {
	return m.Error()
}

// UploadID returns the id of the S3 upload which failed.
func (m multiUploadError) UploadID() string {
	return m.uploadID
}

// UploadInput contains all input for upload requests to Amazon S3.
type UploadInput struct {
	// The canned ACL to apply to the object.
	ACL *string `location:"header" locationName:"x-amz-acl" type:"string"`

	Bucket *string `location:"uri" locationName:"Bucket" type:"string" required:"true"`

	// Specifies caching behavior along the request/reply chain.
	CacheControl *string `location:"header" locationName:"Cache-Control" type:"string"`

	// Specifies presentational information for the object.
	ContentDisposition *string `location:"header" locationName:"Content-Disposition" type:"string"`

	// Specifies what content encodings have been applied to the object and thus
	// what decoding mechanisms must be applied to obtain the media-type referenced
	// by the Content-Type header field.
	ContentEncoding *string `location:"header" locationName:"Content-Encoding" type:"string"`

	// The language the content is in.
	ContentLanguage *string `location:"header" locationName:"Content-Language" type:"string"`

	// A standard MIME type describing the format of the object data.
	ContentType *string `location:"header" locationName:"Content-Type" type:"string"`

	// The date and time at which the object is no longer cacheable.
	Expires *time.Time `location:"header" locationName:"Expires" type:"timestamp" timestampFormat:"rfc822"`

	// Gives the grantee READ, READ_ACP, and WRITE_ACP permissions on the object.
	GrantFullControl *string `location:"header" locationName:"x-amz-grant-full-control" type:"string"`

	// Allows grantee to read the object data and its metadata.
	GrantRead *string `location:"header" locationName:"x-amz-grant-read" type:"string"`

	// Allows grantee to read the object ACL.
	GrantReadACP *string `location:"header" locationName:"x-amz-grant-read-acp" type:"string"`

	// Allows grantee to write the ACL for the applicable object.
	GrantWriteACP *string `location:"header" locationName:"x-amz-grant-write-acp" type:"string"`

	Key *string `location:"uri" locationName:"Key" type:"string" required:"true"`

	// A map of metadata to store with the object in S3.
	Metadata map[string]*string `location:"headers" locationName:"x-amz-meta-" type:"map"`

	// Confirms that the requester knows that she or he will be charged for the
	// request. Bucket owners need not specify this parameter in their requests.
	// Documentation on downloading objects from requester pays buckets can be found
	// at http://docs.aws.amazon.com/AmazonS3/latest/dev/ObjectsinRequesterPaysBuckets.html
	RequestPayer *string `location:"header" locationName:"x-amz-request-payer" type:"string"`

	// Specifies the algorithm to use to when encrypting the object (e.g., AES256,
	// aws:kms).
	SSECustomerAlgorithm *string `location:"header" locationName:"x-amz-server-side-encryption-customer-algorithm" type:"string"`

	// Specifies the customer-provided encryption key for Amazon S3 to use in encrypting
	// data. This value is used to store the object and then it is discarded; Amazon
	// does not store the encryption key. The key must be appropriate for use with
	// the algorithm specified in the x-amz-server-side​-encryption​-customer-algorithm
	// header.
	SSECustomerKey *string `location:"header" locationName:"x-amz-server-side-encryption-customer-key" type:"string"`

	// Specifies the 128-bit MD5 digest of the encryption key according to RFC 1321.
	// Amazon S3 uses this header for a message integrity check to ensure the encryption
	// key was transmitted without error.
	SSECustomerKeyMD5 *string `location:"header" locationName:"x-amz-server-side-encryption-customer-key-MD5" type:"string"`

	// Specifies the AWS KMS key ID to use for object encryption. All GET and PUT
	// requests for an object protected by AWS KMS will fail if not made via SSL
	// or using SigV4. Documentation on configuring any of the officially supported
	// AWS SDKs and CLI can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingAWSSDK.html#specify-signature-version
	SSEKMSKeyID *string `location:"header" locationName:"x-amz-server-side-encryption-aws-kms-key-id" type:"string"`

	// The Server-side encryption algorithm used when storing this object in S3
	// (e.g., AES256, aws:kms).
	ServerSideEncryption *string `location:"header" locationName:"x-amz-server-side-encryption" type:"string"`

	// The type of storage to use for the object. Defaults to 'STANDARD'.
	StorageClass *string `location:"header" locationName:"x-amz-storage-class" type:"string"`

	// If the bucket is configured as a website, redirects requests for this object
	// to another object in the same bucket or to an external URL. Amazon S3 stores
	// the value of this header in the object metadata.
	WebsiteRedirectLocation *string `location:"header" locationName:"x-amz-website-redirect-location" type:"string"`

	// The readable body payload to send to S3.
	Body io.Reader
}

// UploadOutput represents a response from the Upload() call.
type UploadOutput struct {
	// The URL where the object was uploaded to.
	Location string

	// The version of the object that was uploaded. Will only be populated if
	// the S3 Bucket is versioned. If the bucket is not versioned this field
	// will not be set.
	VersionID *string

	// The ID for a multipart upload to S3. In the case of an error the error
	// can be cast to the MultiUploadFailure interface to extract the upload ID.
	UploadID string
}

// The Uploader structure that calls Upload(). It is safe to call Upload()
// on this structure for multiple objects and across concurrent goroutines.
// Mutating the Uploader's properties is not safe to be done concurrently.
type Uploader struct {
	// The buffer size (in bytes) to use when buffering data into chunks and
	// sending them as parts to S3. The minimum allowed part size is 5MB, and
	// if this value is set to zero, the DefaultPartSize value will be used.
	PartSize int64

	// The number of goroutines to spin up in parallel when sending parts.
	// If this is set to zero, the DefaultConcurrency value will be used.
	Concurrency int

	// Setting this value to true will cause the SDK to avoid calling
	// AbortMultipartUpload on a failure, leaving all successfully uploaded
	// parts on S3 for manual recovery.
	//
	// Note that storing parts of an incomplete multipart upload counts towards
	// space usage on S3 and will add additional costs if not cleaned up.
	LeavePartsOnError bool

	// MaxUploadParts is the max number of parts which will be uploaded to S3.
	// Will be used to calculate the partsize of the object to be uploaded.
	// E.g: 5GB file, with MaxUploadParts set to 100, will upload the file
	// as 100, 50MB parts.
	// With a limited of s3.MaxUploadParts (10,000 parts).
	MaxUploadParts int

	// The client to use when uploading to S3.
	S3 s3iface.S3API
}

// NewUploader creates a new Uploader instance to upload objects to S3. Pass In
// additional functional options to customize the uploader's behavior. Requires a
// client.ConfigProvider in order to create a S3 service client. The session.Session
// satisfies the client.ConfigProvider interface.
//
// Example:
//     // The session the S3 Uploader will use
//     sess := session.New()
//
//     // Create an uploader with the session and default options
//     uploader := s3manager.NewUploader(sess)
//
//     // Create an uploader with the session and custom options
//     uploader := s3manager.NewUploader(session, func(u *s3manager.Uploader) {
//          u.PartSize = 64 * 1024 * 1024 // 64MB per part
//     })
func NewUploader(c client.ConfigProvider, options ...func(*Uploader)) *Uploader {
	u := &Uploader{
		S3:                s3.New(c),
		PartSize:          DefaultUploadPartSize,
		Concurrency:       DefaultUploadConcurrency,
		LeavePartsOnError: false,
		MaxUploadParts:    MaxUploadParts,
	}

	for _, option := range options {
		option(u)
	}

	return u
}

// NewUploaderWithClient creates a new Uploader instance to upload objects to S3. Pass in
// additional functional options to customize the uploader's behavior. Requires
// a S3 service client to make S3 API calls.
//
// Example:
//     // S3 service client the Upload manager will use.
//     s3Svc := s3.New(session.New())
//
//     // Create an uploader with S3 client and default options
//     uploader := s3manager.NewUploaderWithClient(s3Svc)
//
//     // Create an uploader with S3 client and custom options
//     uploader := s3manager.NewUploaderWithClient(s3Svc, func(u *s3manager.Uploader) {
//          u.PartSize = 64 * 1024 * 1024 // 64MB per part
//     })
func NewUploaderWithClient(svc s3iface.S3API, options ...func(*Uploader)) *Uploader {
	u := &Uploader{
		S3:                svc,
		PartSize:          DefaultUploadPartSize,
		Concurrency:       DefaultUploadConcurrency,
		LeavePartsOnError: false,
		MaxUploadParts:    MaxUploadParts,
	}

	for _, option := range options {
		option(u)
	}

	return u
}

// Upload uploads an object to S3, intelligently buffering large files into
// smaller chunks and sending them in parallel across multiple goroutines. You
// can configure the buffer size and concurrency through the Uploader's parameters.
//
// Additional functional options can be provided to configure the individual
// upload. These options are copies of the Uploader instance Upload is called from.
// Modifying the options will not impact the original Uploader instance.
//
// It is safe to call this method concurrently across goroutines.
//
// Example:
//     // Upload input parameters
//     upParams := &s3manager.UploadInput{
//         Bucket: &bucketName,
//         Key:    &keyName,
//         Body:   file,
//     }
//
//     // Perform an upload.
//     result, err := uploader.Upload(upParams)
//
//     // Perform upload with options different than the those in the Uploader.
//     result, err := uploader.Upload(upParams, func(u *s3manager.Uploader) {
//          u.PartSize = 10 * 1024 * 1024 // 10MB part size
//          u.LeavePartsOnError = true    // Dont delete the parts if the upload fails.
//     })
func (u Uploader) Upload(input *UploadInput, options ...func(*Uploader)) (*UploadOutput, error) {
	i := uploader{in: input, ctx: u}

	for _, option := range options {
		option(&i.ctx)
	}

	return i.upload()
}

// internal structure to manage an upload to S3.
type uploader struct {
	ctx Uploader

	in *UploadInput

	readerPos int64 // current reader position
	totalSize int64 // set to -1 if the size is not known
}

// internal logic for deciding whether to upload a single part or use a
// multipart upload.
func (u *uploader) upload() (*UploadOutput, error) {
	u.init()

	if u.ctx.PartSize < MinUploadPartSize {
		msg := fmt.Sprintf("part size must be at least %d bytes", MinUploadPartSize)
		return nil, awserr.New("ConfigError", msg, nil)
	}

	// Do one read to determine if we have more than one part
	buf, err := u.nextReader()
	if err == io.EOF || err == io.ErrUnexpectedEOF { // single part
		return u.singlePart(buf)
	} else if err != nil {
		return nil, awserr.New("ReadRequestBody", "read upload data failed", err)
	}

	mu := multiuploader{uploader: u}
	return mu.upload(buf)
}

// init will initialize all default options.
func (u *uploader) init() {
	if u.ctx.Concurrency == 0 {
		u.ctx.Concurrency = DefaultUploadConcurrency
	}
	if u.ctx.PartSize == 0 {
		u.ctx.PartSize = DefaultUploadPartSize
	}

	// Try to get the total size for some optimizations
	u.initSize()
}

// initSize tries to detect the total stream size, setting u.totalSize. If
// the size is not known, totalSize is set to -1.
func (u *uploader) initSize() {
	u.totalSize = -1

	switch r := u.in.Body.(type) {
	case io.Seeker:
		pos, _ := r.Seek(0, 1)
		defer r.Seek(pos, 0)

		n, err := r.Seek(0, 2)
		if err != nil {
			return
		}
		u.totalSize = n

		// Try to adjust partSize if it is too small and account for
		// integer division truncation.
		if u.totalSize/u.ctx.PartSize >= int64(u.ctx.MaxUploadParts) {
			// Add one to the part size to account for remainders
			// during the size calculation. e.g odd number of bytes.
			u.ctx.PartSize = (u.totalSize / int64(u.ctx.MaxUploadParts)) + 1
		}
	}
}

// nextReader returns a seekable reader representing the next packet of data.
// This operation increases the shared u.readerPos counter, but note that it
// does not need to be wrapped in a mutex because nextReader is only called
// from the main thread.
func (u *uploader) nextReader() (io.ReadSeeker, error) {
	switch r := u.in.Body.(type) {
	case io.ReaderAt:
		var err error

		n := u.ctx.PartSize
		if u.totalSize >= 0 {
			bytesLeft := u.totalSize - u.readerPos

			if bytesLeft == 0 {
				err = io.EOF
				n = bytesLeft
			} else if bytesLeft <= u.ctx.PartSize {
				err = io.ErrUnexpectedEOF
				n = bytesLeft
			}
		}

		buf := io.NewSectionReader(r, u.readerPos, n)
		u.readerPos += n

		return buf, err

	default:
		packet := make([]byte, u.ctx.PartSize)
		n, err := io.ReadFull(u.in.Body, packet)
		u.readerPos += int64(n)

		return bytes.NewReader(packet[0:n]), err
	}
}

// singlePart contains upload logic for uploading a single chunk via
// a regular PutObject request. Multipart requests require at least two
// parts, or at least 5MB of data.
func (u *uploader) singlePart(buf io.ReadSeeker) (*UploadOutput, error) {
	params := &s3.PutObjectInput{}
	awsutil.Copy(params, u.in)
	params.Body = buf

	req, out := u.ctx.S3.PutObjectRequest(params)
	req.Handlers.Build.PushBack(request.MakeAddToUserAgentFreeFormHandler("S3Manager"))
	if err := req.Send(); err != nil {
		return nil, err
	}

	url := req.HTTPRequest.URL.String()
	return &UploadOutput{
		Location:  url,
		VersionID: out.VersionId,
	}, nil
}

// internal structure to manage a specific multipart upload to S3.
type multiuploader struct {
	*uploader
	wg       sync.WaitGroup
	m        sync.Mutex
	err      error
	uploadID string
	parts    completedParts
}

// keeps track of a single chunk of data being sent to S3.
type chunk struct {
	buf io.ReadSeeker
	num int64
}

// completedParts is a wrapper to make parts sortable by their part number,
// since S3 required this list to be sent in sorted order.
type completedParts []*s3.CompletedPart

func (a completedParts) Len() int           { return len(a) }
func (a completedParts) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a completedParts) Less(i, j int) bool { return *a[i].PartNumber < *a[j].PartNumber }

// upload will perform a multipart upload using the firstBuf buffer containing
// the first chunk of data.
func (u *multiuploader) upload(firstBuf io.ReadSeeker) (*UploadOutput, error) {
	params := &s3.CreateMultipartUploadInput{}
	awsutil.Copy(params, u.in)

	// Create the multipart
	req, resp := u.ctx.S3.CreateMultipartUploadRequest(params)
	req.Handlers.Build.PushBack(request.MakeAddToUserAgentFreeFormHandler("S3Manager"))
	if err := req.Send(); err != nil {
		return nil, err
	}
	u.uploadID = *resp.UploadId

	// Create the workers
	ch := make(chan chunk, u.ctx.Concurrency)
	for i := 0; i < u.ctx.Concurrency; i++ {
		u.wg.Add(1)
		go u.readChunk(ch)
	}

	// Send part 1 to the workers
	var num int64 = 1
	ch <- chunk{buf: firstBuf, num: num}

	// Read and queue the rest of the parts
	for u.geterr() == nil {
		// This upload exceeded maximum number of supported parts, error now.
		if num > int64(u.ctx.MaxUploadParts) || num > int64(MaxUploadParts) {
			var msg string
			if num > int64(u.ctx.MaxUploadParts) {
				msg = fmt.Sprintf("exceeded total allowed configured MaxUploadParts (%d). Adjust PartSize to fit in this limit",
					u.ctx.MaxUploadParts)
			} else {
				msg = fmt.Sprintf("exceeded total allowed S3 limit MaxUploadParts (%d). Adjust PartSize to fit in this limit",
					MaxUploadParts)
			}
			u.seterr(awserr.New("TotalPartsExceeded", msg, nil))
			break
		}
		num++

		buf, err := u.nextReader()
		if err == io.EOF {
			break
		}

		ch <- chunk{buf: buf, num: num}

		if err != nil && err != io.ErrUnexpectedEOF {
			u.seterr(awserr.New(
				"ReadRequestBody",
				"read multipart upload data failed",
				err))
			break
		}
	}

	// Close the channel, wait for workers, and complete upload
	close(ch)
	u.wg.Wait()
	complete := u.complete()

	if err := u.geterr(); err != nil {
		return nil, &multiUploadError{
			awsError: awserr.New(
				"MultipartUpload",
				"upload multipart failed",
				err),
			uploadID: u.uploadID,
		}
	}
	return &UploadOutput{
		Location:  *complete.Location,
		VersionID: complete.VersionId,
		UploadID:  u.uploadID,
	}, nil
}

// readChunk runs in worker goroutines to pull chunks off of the ch channel
// and send() them as UploadPart requests.
func (u *multiuploader) readChunk(ch chan chunk) {
	defer u.wg.Done()
	for {
		data, ok := <-ch

		if !ok {
			break
		}

		if u.geterr() == nil {
			if err := u.send(data); err != nil {
				u.seterr(err)
			}
		}
	}
}

// send performs an UploadPart request and keeps track of the completed
// part information.
func (u *multiuploader) send(c chunk) error {
	req, resp := u.ctx.S3.UploadPartRequest(&s3.UploadPartInput{
		Bucket:     u.in.Bucket,
		Key:        u.in.Key,
		Body:       c.buf,
		UploadId:   &u.uploadID,
		PartNumber: &c.num,
	})
	req.Handlers.Build.PushBack(request.MakeAddToUserAgentFreeFormHandler("S3Manager"))
	if err := req.Send(); err != nil {
		return err
	}

	n := c.num
	completed := &s3.CompletedPart{ETag: resp.ETag, PartNumber: &n}

	u.m.Lock()
	u.parts = append(u.parts, completed)
	u.m.Unlock()

	return nil
}

// geterr is a thread-safe getter for the error object
func (u *multiuploader) geterr() error {
	u.m.Lock()
	defer u.m.Unlock()

	return u.err
}

// seterr is a thread-safe setter for the error object
func (u *multiuploader) seterr(e error) {
	u.m.Lock()
	defer u.m.Unlock()

	u.err = e
}

// fail will abort the multipart unless LeavePartsOnError is set to true.
func (u *multiuploader) fail() {
	if u.ctx.LeavePartsOnError {
		return
	}

	req, _ := u.ctx.S3.AbortMultipartUploadRequest(&s3.AbortMultipartUploadInput{
		Bucket:   u.in.Bucket,
		Key:      u.in.Key,
		UploadId: &u.uploadID,
	})
	req.Handlers.Build.PushBack(request.MakeAddToUserAgentFreeFormHandler("S3Manager"))
	req.Send()
}

// complete successfully completes a multipart upload and returns the response.
func (u *multiuploader) complete() *s3.CompleteMultipartUploadOutput {
	if u.geterr() != nil {
		u.fail()
		return nil
	}

	// Parts must be sorted in PartNumber order.
	sort.Sort(u.parts)

	req, resp := u.ctx.S3.CompleteMultipartUploadRequest(&s3.CompleteMultipartUploadInput{
		Bucket:          u.in.Bucket,
		Key:             u.in.Key,
		UploadId:        &u.uploadID,
		MultipartUpload: &s3.CompletedMultipartUpload{Parts: u.parts},
	})
	req.Handlers.Build.PushBack(request.MakeAddToUserAgentFreeFormHandler("S3Manager"))
	if err := req.Send(); err != nil {
		u.seterr(err)
		u.fail()
	}

	return resp
}
