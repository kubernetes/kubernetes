// +build go1.13,integration,perftest

package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

var config Config

func main() {
	parseCommandLine()

	log.SetOutput(os.Stderr)

	config.Profiler.Start()
	defer config.Profiler.Stop()

	var err error
	key := config.Key
	size := config.Size
	if len(key) == 0 {
		uploadPartSize := getUploadPartSize(size, config.UploadPartSize, config.SDK.PartSize)
		log.Printf("uploading %s file to s3://%s\n", integration.SizeToName(int(config.Size)), config.Bucket)
		key, err = setupDownloadTest(config.Bucket, config.Size, uploadPartSize)
		if err != nil {
			log.Fatalf("failed to setup download testing: %v", err)
		}

		defer func() {
			log.Printf("cleaning up s3://%s/%s\n", config.Bucket, key)
			if err = teardownDownloadTest(config.Bucket, key); err != nil {
				log.Fatalf("failed to teardwn test artifacts: %v", err)
			}
		}()
	} else {
		size, err = getObjectSize(config.Bucket, key)
		if err != nil {
			log.Fatalf("failed to get object size: %v", err)
		}
	}

	traces := make(chan *RequestTrace, config.SDK.Concurrency)
	requestTracer := downloadRequestTracer(traces)
	downloader := newDownloader(config.Client, config.SDK, requestTracer)

	metricReportDone := startTraceReceiver(traces)

	log.Println("starting download...")
	start := time.Now()
	_, err = downloader.Download(&awstesting.DiscardAt{}, &s3.GetObjectInput{
		Bucket: &config.Bucket,
		Key:    &key,
	})
	if err != nil {
		log.Fatalf("failed to download object, %v", err)
	}
	close(traces)

	dur := time.Since(start)
	log.Printf("Download finished, Size: %d, Dur: %s, Throughput: %.5f GB/s",
		size, dur, (float64(size)/(float64(dur)/float64(time.Second)))/float64(1e9),
	)

	<-metricReportDone
}

func parseCommandLine() {
	config.SetupFlags("", flag.CommandLine)

	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		flag.CommandLine.PrintDefaults()
		log.Fatalf("failed to parse CLI commands")
	}
	if err := config.Validate(); err != nil {
		flag.CommandLine.PrintDefaults()
		log.Fatalf("invalid arguments: %v", err)
	}
}

func setupDownloadTest(bucket string, fileSize, partSize int64) (key string, err error) {
	er := &awstesting.EndlessReader{}
	lr := io.LimitReader(er, fileSize)

	key = integration.UniqueID()

	sess := session.Must(session.NewSession(&aws.Config{
		S3DisableContentMD5Validation: aws.Bool(true),
		S3Disable100Continue:          aws.Bool(true),
	}))

	uploader := s3manager.NewUploader(sess, func(u *s3manager.Uploader) {
		u.PartSize = partSize
		u.Concurrency = runtime.NumCPU() * 2
		u.RequestOptions = append(u.RequestOptions, func(r *request.Request) {
			if r.Operation.Name != "UploadPart" && r.Operation.Name != "PutObject" {
				return
			}

			r.HTTPRequest.Header.Set("X-Amz-Content-Sha256", "UNSIGNED-PAYLOAD")
		})
	})

	_, err = uploader.Upload(&s3manager.UploadInput{
		Bucket: &bucket,
		Body:   lr,
		Key:    &key,
	})
	if err != nil {
		err = fmt.Errorf("failed to upload test object to s3: %v", err)
	}

	return
}

func teardownDownloadTest(bucket, key string) error {
	sess := session.Must(session.NewSession())

	svc := s3.New(sess)

	_, err := svc.DeleteObject(&s3.DeleteObjectInput{Bucket: &bucket, Key: &key})
	return err
}

func startTraceReceiver(traces <-chan *RequestTrace) <-chan struct{} {
	metricReportDone := make(chan struct{})

	go func() {
		defer close(metricReportDone)
		metrics := map[string]*RequestTrace{}
		for trace := range traces {
			curTrace, ok := metrics[trace.Operation]
			if !ok {
				curTrace = trace
			} else {
				curTrace.attempts = append(curTrace.attempts, trace.attempts...)
				if len(trace.errs) != 0 {
					curTrace.errs = append(curTrace.errs, trace.errs...)
				}
				curTrace.finish = trace.finish
			}

			metrics[trace.Operation] = curTrace
		}

		for _, name := range []string{
			"GetObject",
		} {
			if trace, ok := metrics[name]; ok {
				printAttempts(name, trace, config.LogVerbose)
			}
		}
	}()

	return metricReportDone
}

func printAttempts(op string, trace *RequestTrace, verbose bool) {
	if !verbose {
		return
	}

	log.Printf("%s: latency:%s requests:%d errors:%d",
		op,
		trace.finish.Sub(trace.start),
		len(trace.attempts),
		len(trace.errs),
	)

	for _, a := range trace.attempts {
		log.Printf("  * %s", a)
	}
	if err := trace.Err(); err != nil {
		log.Printf("Operation Errors: %v", err)
	}
	log.Println()
}

func downloadRequestTracer(traces chan<- *RequestTrace) request.Option {
	tracerOption := func(r *request.Request) {
		id := "op"
		if v, ok := r.Params.(*s3.GetObjectInput); ok {
			if v.Range != nil {
				id = *v.Range
			}
		}
		tracer := NewRequestTrace(r.Context(), r.Operation.Name, id)
		r.SetContext(tracer)

		r.Handlers.Send.PushFront(tracer.OnSendAttempt)
		r.Handlers.CompleteAttempt.PushBack(tracer.OnCompleteAttempt)
		r.Handlers.Complete.PushBack(tracer.OnComplete)
		r.Handlers.Complete.PushBack(func(rr *request.Request) {
			traces <- tracer
		})
	}

	return tracerOption
}

func newDownloader(clientConfig ClientConfig, sdkConfig SDKConfig, options ...request.Option) *s3manager.Downloader {
	client := NewClient(clientConfig)

	sess, err := session.NewSessionWithOptions(session.Options{
		Config:            aws.Config{HTTPClient: client},
		SharedConfigState: session.SharedConfigEnable,
	})
	if err != nil {
		log.Fatalf("failed to load session, %v", err)
	}

	downloader := s3manager.NewDownloader(sess, func(d *s3manager.Downloader) {
		d.PartSize = sdkConfig.PartSize
		d.Concurrency = sdkConfig.Concurrency
		d.BufferProvider = sdkConfig.BufferProvider

		d.RequestOptions = append(d.RequestOptions, options...)
	})

	return downloader
}

func getObjectSize(bucket, key string) (int64, error) {
	sess := session.Must(session.NewSession())
	svc := s3.New(sess)
	resp, err := svc.HeadObject(&s3.HeadObjectInput{
		Bucket: &bucket,
		Key:    &key,
	})
	if err != nil {
		return 0, err
	}

	return *resp.ContentLength, nil
}

type Profiler struct {
	outputDir string

	enableCPU   bool
	enableTrace bool

	cpuFile   *os.File
	traceFile *os.File
}

func (p *Profiler) SetupFlags(prefix string, flagSet *flag.FlagSet) {
	prefix += "profiler."

	flagSet.StringVar(&p.outputDir, prefix+"output-dir", os.TempDir(), "output directory to write profiling data")
	flagSet.BoolVar(&p.enableCPU, prefix+"cpu", false, "enable CPU profiling")
	flagSet.BoolVar(&p.enableTrace, prefix+"trace", false, "enable tracing")
}

func (p *Profiler) Start() {
	var err error

	uuid := integration.UniqueID()
	if p.enableCPU {
		p.cpuFile, err = p.createFile(uuid, "cpu")
		if err != nil {
			panic(fmt.Sprintf("failed to create cpu profile file: %v", err))
		}
		err = pprof.StartCPUProfile(p.cpuFile)
		if err != nil {
			panic(fmt.Sprintf("failed to start cpu profile: %v", err))
		}
	}
	if p.enableTrace {
		p.traceFile, err = p.createFile(uuid, "trace")
		if err != nil {
			panic(fmt.Sprintf("failed to create trace file: %v", err))
		}
		err = trace.Start(p.traceFile)
		if err != nil {
			panic(fmt.Sprintf("failed to tracing: %v", err))
		}
	}
}

func (p *Profiler) logAndCloseFile(profile string, file *os.File) {
	info, err := file.Stat()
	if err != nil {
		log.Printf("failed to stat %s profile: %v", profile, err)
	} else {
		log.Printf("writing %s profile to: %v", profile, filepath.Join(p.outputDir, info.Name()))
	}
	file.Close()
}

func (p *Profiler) Stop() {
	if p.enableCPU {
		pprof.StopCPUProfile()
		p.logAndCloseFile("cpu", p.cpuFile)
	}
	if p.enableTrace {
		trace.Stop()
		p.logAndCloseFile("trace", p.traceFile)
	}
}

func (p *Profiler) createFile(prefix, name string) (*os.File, error) {
	return os.OpenFile(filepath.Join(p.outputDir, prefix+"."+name+".profile"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0666)
}

func getUploadPartSize(fileSize, uploadPartSize, downloadPartSize int64) int64 {
	partSize := uploadPartSize

	if partSize == 0 {
		partSize = downloadPartSize
	}
	if fileSize/partSize > s3manager.MaxUploadParts {
		partSize = (fileSize / s3manager.MaxUploadParts) + 1
	}

	return partSize
}
