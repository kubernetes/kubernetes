// +build integration,perftest

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/signal"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

var config Config

func init() {
	config.SetupFlags("", flag.CommandLine)
}

func main() {
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		flag.CommandLine.PrintDefaults()
		exitErrorf(err, "failed to parse CLI commands")
	}
	if err := config.Validate(); err != nil {
		flag.CommandLine.PrintDefaults()
		exitErrorf(err, "invalid arguments")
	}

	client := NewClient(config.Client)

	var creds *credentials.Credentials
	if config.SDK.Anonymous {
		creds = credentials.AnonymousCredentials
	}

	var endpoint *string
	if v := config.Endpoint; len(v) != 0 {
		endpoint = &v
	}

	sess, err := session.NewSession(&aws.Config{
		HTTPClient:           client,
		Endpoint:             endpoint,
		Credentials:          creds,
		S3Disable100Continue: aws.Bool(!config.SDK.ExpectContinue),
	})
	if err != nil {
		exitErrorf(err, "failed to load config")
	}

	// Create context cancel for Ctrl+C/Interrupt
	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	go func() {
		<-sigCh
		cancelFn()
	}()

	// Use the request duration timeout if specified.
	if config.RequestDuration != 0 {
		var timeoutFn func()
		ctx, timeoutFn = context.WithTimeout(ctx, config.RequestDuration)
		defer timeoutFn()
	}

	logger := NewLogger(os.Stdout)

	// Start making the requests.
	svc := s3.New(sess)
	var reqCount int64
	errCount := 0
	for {
		trace := doRequest(ctx, reqCount, svc, config)
		select {
		case <-ctx.Done():
			return
		default:
		}
		logger.RecordTrace(trace)

		if err := trace.Err(); err != nil {
			fmt.Fprintf(os.Stderr, err.Error())
			errCount++
		} else {
			errCount = 0
		}

		if config.RequestCount > 0 && reqCount == config.RequestCount {
			return
		}

		reqCount++

		// If the first several requests fail, exist, something is broken.
		if errCount == 5 && reqCount == 5 {
			exitErrorf(trace.Err(), "unable to make requests")
		}

		if config.RequestDelay > 0 {
			time.Sleep(config.RequestDelay)
		}
	}
}

func doRequest(ctx context.Context, id int64, svc *s3.S3, config Config) *RequestTrace {
	traceCtx := NewRequestTrace(ctx, id)
	defer traceCtx.RequestDone()

	resp, err := svc.GetObjectWithContext(traceCtx, &s3.GetObjectInput{
		Bucket: &config.Bucket,
		Key:    &config.Key,
	}, func(r *request.Request) {
		r.Handlers.Send.PushFront(traceCtx.OnSendAttempt)
		r.Handlers.Complete.PushBack(traceCtx.OnCompleteRequest)
		r.Handlers.CompleteAttempt.PushBack(traceCtx.OnCompleteAttempt)
	})
	if err != nil {
		traceCtx.AppendError(fmt.Errorf("request failed, %v", err))
		return traceCtx
	}
	defer resp.Body.Close()

	if n, err := io.Copy(ioutil.Discard, resp.Body); err != nil {
		traceCtx.AppendError(fmt.Errorf("read request body failed, read %v, %v", n, err))
		return traceCtx
	}

	return traceCtx
}

func exitErrorf(err error, msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "FAILED: %v\n"+msg+"\n", append([]interface{}{err}, args...)...)
	os.Exit(1)
}
