// +build integration,perftest

package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/internal/sdkio"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

var benchConfig BenchmarkConfig

type BenchmarkConfig struct {
	bucket       string
	tempdir      string
	clientConfig ClientConfig
	sizes        string
	parts        string
	concurrency  string
	bufferSize   string
}

func (b *BenchmarkConfig) SetupFlags(prefix string, flagSet *flag.FlagSet) {
	flagSet.StringVar(&b.bucket, "bucket", "", "Bucket to use for benchmark")
	flagSet.StringVar(&b.tempdir, "temp", os.TempDir(), "location to create temporary files")

	flagSet.StringVar(&b.sizes, "size",
		fmt.Sprintf("%d,%d",
			5*sdkio.MebiByte,
			1*sdkio.GibiByte), "file sizes to benchmark separated by comma")

	flagSet.StringVar(&b.parts, "part",
		fmt.Sprintf("%d,%d,%d",
			s3manager.DefaultUploadPartSize,
			25*sdkio.MebiByte,
			100*sdkio.MebiByte), "part sizes to benchmark separated by comma")

	flagSet.StringVar(&b.concurrency, "concurrency",
		fmt.Sprintf("%d,%d,%d",
			s3manager.DefaultUploadConcurrency,
			2*s3manager.DefaultUploadConcurrency,
			100),
		"concurrences to benchmark separated comma")

	flagSet.StringVar(&b.bufferSize, "buffer", fmt.Sprintf("%d,%d", 0, 1*sdkio.MebiByte), "part sizes to benchmark separated comma")
	b.clientConfig.SetupFlags(prefix, flagSet)
}

func (b *BenchmarkConfig) BufferSizes() []int {
	ints, err := b.stringToInt(b.bufferSize)
	if err != nil {
		panic(fmt.Sprintf("failed to parse file sizes: %v", err))
	}

	return ints
}

func (b *BenchmarkConfig) FileSizes() []int64 {
	ints, err := b.stringToInt64(b.sizes)
	if err != nil {
		panic(fmt.Sprintf("failed to parse file sizes: %v", err))
	}

	return ints
}

func (b *BenchmarkConfig) PartSizes() []int64 {
	ints, err := b.stringToInt64(b.parts)
	if err != nil {
		panic(fmt.Sprintf("failed to parse part sizes: %v", err))
	}

	return ints
}

func (b *BenchmarkConfig) Concurrences() []int {
	ints, err := b.stringToInt(b.concurrency)
	if err != nil {
		panic(fmt.Sprintf("failed to parse part sizes: %v", err))
	}

	return ints
}

func (b *BenchmarkConfig) stringToInt(s string) ([]int, error) {
	int64s, err := b.stringToInt64(s)
	if err != nil {
		return nil, err
	}

	var ints []int
	for i := range int64s {
		ints = append(ints, int(int64s[i]))
	}

	return ints, nil
}

func (b *BenchmarkConfig) stringToInt64(s string) ([]int64, error) {
	var sizes []int64

	split := strings.Split(s, ",")

	for _, size := range split {
		size = strings.Trim(size, " ")
		i, err := strconv.ParseInt(size, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %s: %v", size, err)
		}

		sizes = append(sizes, i)
	}

	return sizes, nil
}

func BenchmarkUpload(b *testing.B) {
	baseSdkConfig := SDKConfig{WithUnsignedPayload: true}

	for _, fileSize := range benchConfig.FileSizes() {
		b.Run(fmt.Sprintf("%s File", integration.SizeToName(int(fileSize))), func(b *testing.B) {
			for _, concurrency := range benchConfig.Concurrences() {
				b.Run(fmt.Sprintf("%d Concurrency", concurrency), func(b *testing.B) {
					for _, partSize := range benchConfig.PartSizes() {
						if partSize > fileSize {
							continue
						}
						partSize = getUploadPartSize(fileSize, partSize)
						b.Run(fmt.Sprintf("%s PartSize", integration.SizeToName(int(partSize))), func(b *testing.B) {
							for _, bufferSize := range benchConfig.BufferSizes() {
								var name string
								if bufferSize == 0 {
									name = "Unbuffered"
								} else {
									name = fmt.Sprintf("%s Buffer", integration.SizeToName(bufferSize))
								}
								b.Run(name, func(b *testing.B) {
									sdkConfig := baseSdkConfig

									sdkConfig.Concurrency = concurrency
									sdkConfig.PartSize = partSize
									if bufferSize > 0 {
										sdkConfig.BufferProvider = s3manager.NewBufferedReadSeekerWriteToPool(bufferSize)
									}

									for i := 0; i < b.N; i++ {
										for {
											b.ResetTimer()
											reader := aws.ReadSeekCloser(io.LimitReader(&awstesting.EndlessReader{}, fileSize))
											err := benchUpload(b, benchConfig.bucket, integration.UniqueID(), reader, sdkConfig, benchConfig.clientConfig)
											if err != nil {
												b.Logf("upload failed, retrying: %v", err)
												continue
											}
											break
										}
									}
								})
							}
						})
					}
				})
			}
		})
	}
}

func benchUpload(b *testing.B, bucket, key string, reader io.ReadSeeker, sdkConfig SDKConfig, clientConfig ClientConfig) error {
	uploader := newUploader(clientConfig, sdkConfig, SetUnsignedPayload)
	_, err := uploader.Upload(&s3manager.UploadInput{
		Bucket: &bucket,
		Key:    &key,
		Body:   reader,
	})
	if err != nil {
		return err
	}
	return nil
}

func TestMain(m *testing.M) {
	benchConfig.SetupFlags("", flag.CommandLine)
	flag.Parse()
	os.Exit(m.Run())
}
