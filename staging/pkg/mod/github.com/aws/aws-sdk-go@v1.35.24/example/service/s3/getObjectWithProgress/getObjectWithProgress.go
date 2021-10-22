// +build example

package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"sync/atomic"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

// progressWriter tracks the download progress of a file from S3 to a file
// as the writeAt method is called, the byte size is added to the written total,
// and then a log is printed of the written percentage from the total size
// it looks like this on the command line:
//  2019/02/22 12:59:15 File size:35943530 downloaded:16360 percentage:0%
//  2019/02/22 12:59:15 File size:35943530 downloaded:16988 percentage:0%
//  2019/02/22 12:59:15 File size:35943530 downloaded:33348 percentage:0%
type progressWriter struct {
	written int64
	writer  io.WriterAt
	size    int64
}

func (pw *progressWriter) WriteAt(p []byte, off int64) (int, error) {
	atomic.AddInt64(&pw.written, int64(len(p)))

	percentageDownloaded := float32(pw.written*100) / float32(pw.size)

	fmt.Printf("File size:%d downloaded:%d percentage:%.2f%%\r", pw.size, pw.written, percentageDownloaded)

	return pw.writer.WriteAt(p, off)
}

func byteCountDecimal(b int64) string {
	const unit = 1000
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "kMGTPE"[exp])
}

func getFileSize(svc *s3.S3, bucket string, prefix string) (filesize int64, error error) {
	params := &s3.HeadObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(prefix),
	}

	resp, err := svc.HeadObject(params)
	if err != nil {
		return 0, err
	}

	return *resp.ContentLength, nil
}

func parseFilename(keyString string) (filename string) {
	ss := strings.Split(keyString, "/")
	s := ss[len(ss)-1]
	return s
}

func main() {
	if len(os.Args) < 2 {
		log.Println("USAGE ERROR: AWS_REGION=us-east-1 go run getObjWithProgress.go bucket-name object-key")
		return
	}

	bucket := os.Args[1]
	key := os.Args[2]

	filename := parseFilename(key)

	sess, err := session.NewSession()
	if err != nil {
		panic(err)
	}

	s3Client := s3.New(sess)
	downloader := s3manager.NewDownloader(sess)
	size, err := getFileSize(s3Client, bucket, key)
	if err != nil {
		panic(err)
	}

	log.Println("Starting download, size:", byteCountDecimal(size))
	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	temp, err := ioutil.TempFile(cwd, "getObjWithProgress-tmp-")
	if err != nil {
		panic(err)
	}
	tempfileName := temp.Name()

	writer := &progressWriter{writer: temp, size: size, written: 0}
	params := &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	}

	if _, err := downloader.Download(writer, params); err != nil {
		log.Printf("Download failed! Deleting tempfile: %s", tempfileName)
		os.Remove(tempfileName)
		panic(err)
	}

	if err := temp.Close(); err != nil {
		panic(err)
	}

	if err := os.Rename(temp.Name(), filename); err != nil {
		panic(err)
	}

	fmt.Println()
	log.Println("File downloaded! Available at:", filename)
}
