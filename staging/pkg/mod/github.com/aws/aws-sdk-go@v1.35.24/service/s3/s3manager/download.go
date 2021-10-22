package s3manager

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

// DefaultDownloadPartSize is the default range of bytes to get at a time when
// using Download().
const DefaultDownloadPartSize = 1024 * 1024 * 5

// DefaultDownloadConcurrency is the default number of goroutines to spin up
// when using Download().
const DefaultDownloadConcurrency = 5

type errReadingBody struct {
	err error
}

func (e *errReadingBody) Error() string {
	return fmt.Sprintf("failed to read part body: %v", e.err)
}

func (e *errReadingBody) Unwrap() error {
	return e.err
}

// The Downloader structure that calls Download(). It is safe to call Download()
// on this structure for multiple objects and across concurrent goroutines.
// Mutating the Downloader's properties is not safe to be done concurrently.
type Downloader struct {
	// The size (in bytes) to request from S3 for each part.
	// The minimum allowed part size is 5MB, and  if this value is set to zero,
	// the DefaultDownloadPartSize value will be used.
	//
	// PartSize is ignored if the Range input parameter is provided.
	PartSize int64

	// The number of goroutines to spin up in parallel when sending parts.
	// If this is set to zero, the DefaultDownloadConcurrency value will be used.
	//
	// Concurrency of 1 will download the parts sequentially.
	//
	// Concurrency is ignored if the Range input parameter is provided.
	Concurrency int

	// An S3 client to use when performing downloads.
	S3 s3iface.S3API

	// List of request options that will be passed down to individual API
	// operation requests made by the downloader.
	RequestOptions []request.Option

	// Defines the buffer strategy used when downloading a part.
	//
	// If a WriterReadFromProvider is given the Download manager
	// will pass the io.WriterAt of the Download request to the provider
	// and will use the returned WriterReadFrom from the provider as the
	// destination writer when copying from http response body.
	BufferProvider WriterReadFromProvider
}

// WithDownloaderRequestOptions appends to the Downloader's API request options.
func WithDownloaderRequestOptions(opts ...request.Option) func(*Downloader) {
	return func(d *Downloader) {
		d.RequestOptions = append(d.RequestOptions, opts...)
	}
}

// NewDownloader creates a new Downloader instance to downloads objects from
// S3 in concurrent chunks. Pass in additional functional options  to customize
// the downloader behavior. Requires a client.ConfigProvider in order to create
// a S3 service client. The session.Session satisfies the client.ConfigProvider
// interface.
//
// Example:
//     // The session the S3 Downloader will use
//     sess := session.Must(session.NewSession())
//
//     // Create a downloader with the session and default options
//     downloader := s3manager.NewDownloader(sess)
//
//     // Create a downloader with the session and custom options
//     downloader := s3manager.NewDownloader(sess, func(d *s3manager.Downloader) {
//          d.PartSize = 64 * 1024 * 1024 // 64MB per part
//     })
func NewDownloader(c client.ConfigProvider, options ...func(*Downloader)) *Downloader {
	return newDownloader(s3.New(c), options...)
}

func newDownloader(client s3iface.S3API, options ...func(*Downloader)) *Downloader {
	d := &Downloader{
		S3:             client,
		PartSize:       DefaultDownloadPartSize,
		Concurrency:    DefaultDownloadConcurrency,
		BufferProvider: defaultDownloadBufferProvider(),
	}
	for _, option := range options {
		option(d)
	}

	return d
}

// NewDownloaderWithClient creates a new Downloader instance to downloads
// objects from S3 in concurrent chunks. Pass in additional functional
// options to customize the downloader behavior. Requires a S3 service client
// to make S3 API calls.
//
// Example:
//     // The session the S3 Downloader will use
//     sess := session.Must(session.NewSession())
//
//     // The S3 client the S3 Downloader will use
//     s3Svc := s3.New(sess)
//
//     // Create a downloader with the s3 client and default options
//     downloader := s3manager.NewDownloaderWithClient(s3Svc)
//
//     // Create a downloader with the s3 client and custom options
//     downloader := s3manager.NewDownloaderWithClient(s3Svc, func(d *s3manager.Downloader) {
//          d.PartSize = 64 * 1024 * 1024 // 64MB per part
//     })
func NewDownloaderWithClient(svc s3iface.S3API, options ...func(*Downloader)) *Downloader {
	return newDownloader(svc, options...)
}

type maxRetrier interface {
	MaxRetries() int
}

// Download downloads an object in S3 and writes the payload into w using
// concurrent GET requests. The n int64 returned is the size of the object downloaded
// in bytes.
//
// Additional functional options can be provided to configure the individual
// download. These options are copies of the Downloader instance Download is called from.
// Modifying the options will not impact the original Downloader instance.
//
// It is safe to call this method concurrently across goroutines.
//
// The w io.WriterAt can be satisfied by an os.File to do multipart concurrent
// downloads, or in memory []byte wrapper using aws.WriteAtBuffer.
//
// Specifying a Downloader.Concurrency of 1 will cause the Downloader to
// download the parts from S3 sequentially.
//
// If the GetObjectInput's Range value is provided that will cause the downloader
// to perform a single GetObjectInput request for that object's range. This will
// caused the part size, and concurrency configurations to be ignored.
func (d Downloader) Download(w io.WriterAt, input *s3.GetObjectInput, options ...func(*Downloader)) (n int64, err error) {
	return d.DownloadWithContext(aws.BackgroundContext(), w, input, options...)
}

// DownloadWithContext downloads an object in S3 and writes the payload into w
// using concurrent GET requests. The n int64 returned is the size of the object downloaded
// in bytes.
//
// DownloadWithContext is the same as Download with the additional support for
// Context input parameters. The Context must not be nil. A nil Context will
// cause a panic. Use the Context to add deadlining, timeouts, etc. The
// DownloadWithContext may create sub-contexts for individual underlying
// requests.
//
// Additional functional options can be provided to configure the individual
// download. These options are copies of the Downloader instance Download is
// called from. Modifying the options will not impact the original Downloader
// instance. Use the WithDownloaderRequestOptions helper function to pass in request
// options that will be applied to all API operations made with this downloader.
//
// The w io.WriterAt can be satisfied by an os.File to do multipart concurrent
// downloads, or in memory []byte wrapper using aws.WriteAtBuffer.
//
// Specifying a Downloader.Concurrency of 1 will cause the Downloader to
// download the parts from S3 sequentially.
//
// It is safe to call this method concurrently across goroutines.
//
// If the GetObjectInput's Range value is provided that will cause the downloader
// to perform a single GetObjectInput request for that object's range. This will
// caused the part size, and concurrency configurations to be ignored.
func (d Downloader) DownloadWithContext(ctx aws.Context, w io.WriterAt, input *s3.GetObjectInput, options ...func(*Downloader)) (n int64, err error) {
	impl := downloader{w: w, in: input, cfg: d, ctx: ctx}

	for _, option := range options {
		option(&impl.cfg)
	}
	impl.cfg.RequestOptions = append(impl.cfg.RequestOptions, request.WithAppendUserAgent("S3Manager"))

	if s, ok := d.S3.(maxRetrier); ok {
		impl.partBodyMaxRetries = s.MaxRetries()
	}

	impl.totalBytes = -1
	if impl.cfg.Concurrency == 0 {
		impl.cfg.Concurrency = DefaultDownloadConcurrency
	}

	if impl.cfg.PartSize == 0 {
		impl.cfg.PartSize = DefaultDownloadPartSize
	}

	return impl.download()
}

// DownloadWithIterator will download a batched amount of objects in S3 and writes them
// to the io.WriterAt specificed in the iterator.
//
// Example:
//	svc := s3manager.NewDownloader(session)
//
//	fooFile, err := os.Open("/tmp/foo.file")
//	if err != nil {
//		return err
//	}
//
//	barFile, err := os.Open("/tmp/bar.file")
//	if err != nil {
//		return err
//	}
//
//	objects := []s3manager.BatchDownloadObject {
//		{
//			Object: &s3.GetObjectInput {
//				Bucket: aws.String("bucket"),
//				Key: aws.String("foo"),
//			},
//			Writer: fooFile,
//		},
//		{
//			Object: &s3.GetObjectInput {
//				Bucket: aws.String("bucket"),
//				Key: aws.String("bar"),
//			},
//			Writer: barFile,
//		},
//	}
//
//	iter := &s3manager.DownloadObjectsIterator{Objects: objects}
//	if err := svc.DownloadWithIterator(aws.BackgroundContext(), iter); err != nil {
//		return err
//	}
func (d Downloader) DownloadWithIterator(ctx aws.Context, iter BatchDownloadIterator, opts ...func(*Downloader)) error {
	var errs []Error
	for iter.Next() {
		object := iter.DownloadObject()
		if _, err := d.DownloadWithContext(ctx, object.Writer, object.Object, opts...); err != nil {
			errs = append(errs, newError(err, object.Object.Bucket, object.Object.Key))
		}

		if object.After == nil {
			continue
		}

		if err := object.After(); err != nil {
			errs = append(errs, newError(err, object.Object.Bucket, object.Object.Key))
		}
	}

	if len(errs) > 0 {
		return NewBatchError("BatchedDownloadIncomplete", "some objects have failed to download.", errs)
	}
	return nil
}

// downloader is the implementation structure used internally by Downloader.
type downloader struct {
	ctx aws.Context
	cfg Downloader

	in *s3.GetObjectInput
	w  io.WriterAt

	wg sync.WaitGroup
	m  sync.Mutex

	pos        int64
	totalBytes int64
	written    int64
	err        error

	partBodyMaxRetries int
}

// download performs the implementation of the object download across ranged
// GETs.
func (d *downloader) download() (n int64, err error) {
	// If range is specified fall back to single download of that range
	// this enables the functionality of ranged gets with the downloader but
	// at the cost of no multipart downloads.
	if rng := aws.StringValue(d.in.Range); len(rng) > 0 {
		d.downloadRange(rng)
		return d.written, d.err
	}

	// Spin off first worker to check additional header information
	d.getChunk()

	if total := d.getTotalBytes(); total >= 0 {
		// Spin up workers
		ch := make(chan dlchunk, d.cfg.Concurrency)

		for i := 0; i < d.cfg.Concurrency; i++ {
			d.wg.Add(1)
			go d.downloadPart(ch)
		}

		// Assign work
		for d.getErr() == nil {
			if d.pos >= total {
				break // We're finished queuing chunks
			}

			// Queue the next range of bytes to read.
			ch <- dlchunk{w: d.w, start: d.pos, size: d.cfg.PartSize}
			d.pos += d.cfg.PartSize
		}

		// Wait for completion
		close(ch)
		d.wg.Wait()
	} else {
		// Checking if we read anything new
		for d.err == nil {
			d.getChunk()
		}

		// We expect a 416 error letting us know we are done downloading the
		// total bytes. Since we do not know the content's length, this will
		// keep grabbing chunks of data until the range of bytes specified in
		// the request is out of range of the content. Once, this happens, a
		// 416 should occur.
		e, ok := d.err.(awserr.RequestFailure)
		if ok && e.StatusCode() == http.StatusRequestedRangeNotSatisfiable {
			d.err = nil
		}
	}

	// Return error
	return d.written, d.err
}

// downloadPart is an individual goroutine worker reading from the ch channel
// and performing a GetObject request on the data with a given byte range.
//
// If this is the first worker, this operation also resolves the total number
// of bytes to be read so that the worker manager knows when it is finished.
func (d *downloader) downloadPart(ch chan dlchunk) {
	defer d.wg.Done()
	for {
		chunk, ok := <-ch
		if !ok {
			break
		}
		if d.getErr() != nil {
			// Drain the channel if there is an error, to prevent deadlocking
			// of download producer.
			continue
		}

		if err := d.downloadChunk(chunk); err != nil {
			d.setErr(err)
		}
	}
}

// getChunk grabs a chunk of data from the body.
// Not thread safe. Should only used when grabbing data on a single thread.
func (d *downloader) getChunk() {
	if d.getErr() != nil {
		return
	}

	chunk := dlchunk{w: d.w, start: d.pos, size: d.cfg.PartSize}
	d.pos += d.cfg.PartSize

	if err := d.downloadChunk(chunk); err != nil {
		d.setErr(err)
	}
}

// downloadRange downloads an Object given the passed in Byte-Range value.
// The chunk used down download the range will be configured for that range.
func (d *downloader) downloadRange(rng string) {
	if d.getErr() != nil {
		return
	}

	chunk := dlchunk{w: d.w, start: d.pos}
	// Ranges specified will short circuit the multipart download
	chunk.withRange = rng

	if err := d.downloadChunk(chunk); err != nil {
		d.setErr(err)
	}

	// Update the position based on the amount of data received.
	d.pos = d.written
}

// downloadChunk downloads the chunk from s3
func (d *downloader) downloadChunk(chunk dlchunk) error {
	in := &s3.GetObjectInput{}
	awsutil.Copy(in, d.in)

	// Get the next byte range of data
	in.Range = aws.String(chunk.ByteRange())

	var n int64
	var err error
	for retry := 0; retry <= d.partBodyMaxRetries; retry++ {
		n, err = d.tryDownloadChunk(in, &chunk)
		if err == nil {
			break
		}
		// Check if the returned error is an errReadingBody.
		// If err is errReadingBody this indicates that an error
		// occurred while copying the http response body.
		// If this occurs we unwrap the err to set the underlying error
		// and attempt any remaining retries.
		if bodyErr, ok := err.(*errReadingBody); ok {
			err = bodyErr.Unwrap()
		} else {
			return err
		}

		chunk.cur = 0
		logMessage(d.cfg.S3, aws.LogDebugWithRequestRetries,
			fmt.Sprintf("DEBUG: object part body download interrupted %s, err, %v, retrying attempt %d",
				aws.StringValue(in.Key), err, retry))
	}

	d.incrWritten(n)

	return err
}

func (d *downloader) tryDownloadChunk(in *s3.GetObjectInput, w io.Writer) (int64, error) {
	cleanup := func() {}
	if d.cfg.BufferProvider != nil {
		w, cleanup = d.cfg.BufferProvider.GetReadFrom(w)
	}
	defer cleanup()

	resp, err := d.cfg.S3.GetObjectWithContext(d.ctx, in, d.cfg.RequestOptions...)
	if err != nil {
		return 0, err
	}
	d.setTotalBytes(resp) // Set total if not yet set.

	n, err := io.Copy(w, resp.Body)
	resp.Body.Close()
	if err != nil {
		return n, &errReadingBody{err: err}
	}

	return n, nil
}

func logMessage(svc s3iface.S3API, level aws.LogLevelType, msg string) {
	s, ok := svc.(*s3.S3)
	if !ok {
		return
	}

	if s.Config.Logger == nil {
		return
	}

	if s.Config.LogLevel.Matches(level) {
		s.Config.Logger.Log(msg)
	}
}

// getTotalBytes is a thread-safe getter for retrieving the total byte status.
func (d *downloader) getTotalBytes() int64 {
	d.m.Lock()
	defer d.m.Unlock()

	return d.totalBytes
}

// setTotalBytes is a thread-safe setter for setting the total byte status.
// Will extract the object's total bytes from the Content-Range if the file
// will be chunked, or Content-Length. Content-Length is used when the response
// does not include a Content-Range. Meaning the object was not chunked. This
// occurs when the full file fits within the PartSize directive.
func (d *downloader) setTotalBytes(resp *s3.GetObjectOutput) {
	d.m.Lock()
	defer d.m.Unlock()

	if d.totalBytes >= 0 {
		return
	}

	if resp.ContentRange == nil {
		// ContentRange is nil when the full file contents is provided, and
		// is not chunked. Use ContentLength instead.
		if resp.ContentLength != nil {
			d.totalBytes = *resp.ContentLength
			return
		}
	} else {
		parts := strings.Split(*resp.ContentRange, "/")

		total := int64(-1)
		var err error
		// Checking for whether or not a numbered total exists
		// If one does not exist, we will assume the total to be -1, undefined,
		// and sequentially download each chunk until hitting a 416 error
		totalStr := parts[len(parts)-1]
		if totalStr != "*" {
			total, err = strconv.ParseInt(totalStr, 10, 64)
			if err != nil {
				d.err = err
				return
			}
		}

		d.totalBytes = total
	}
}

func (d *downloader) incrWritten(n int64) {
	d.m.Lock()
	defer d.m.Unlock()

	d.written += n
}

// getErr is a thread-safe getter for the error object
func (d *downloader) getErr() error {
	d.m.Lock()
	defer d.m.Unlock()

	return d.err
}

// setErr is a thread-safe setter for the error object
func (d *downloader) setErr(e error) {
	d.m.Lock()
	defer d.m.Unlock()

	d.err = e
}

// dlchunk represents a single chunk of data to write by the worker routine.
// This structure also implements an io.SectionReader style interface for
// io.WriterAt, effectively making it an io.SectionWriter (which does not
// exist).
type dlchunk struct {
	w     io.WriterAt
	start int64
	size  int64
	cur   int64

	// specifies the byte range the chunk should be downloaded with.
	withRange string
}

// Write wraps io.WriterAt for the dlchunk, writing from the dlchunk's start
// position to its end (or EOF).
//
// If a range is specified on the dlchunk the size will be ignored when writing.
// as the total size may not of be known ahead of time.
func (c *dlchunk) Write(p []byte) (n int, err error) {
	if c.cur >= c.size && len(c.withRange) == 0 {
		return 0, io.EOF
	}

	n, err = c.w.WriteAt(p, c.start+c.cur)
	c.cur += int64(n)

	return
}

// ByteRange returns a HTTP Byte-Range header value that should be used by the
// client to request the chunk's range.
func (c *dlchunk) ByteRange() string {
	if len(c.withRange) != 0 {
		return c.withRange
	}

	return fmt.Sprintf("bytes=%d-%d", c.start, c.start+c.size-1)
}
