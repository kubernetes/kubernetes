package getter

import (
	"fmt"
	"net/url"
	"strings"
)

// S3Detector implements Detector to detect S3 URLs and turn
// them into URLs that the S3 getter can understand.
type S3Detector struct{}

func (d *S3Detector) Detect(src, _ string) (string, bool, error) {
	if len(src) == 0 {
		return "", false, nil
	}

	if strings.Contains(src, ".amazonaws.com/") {
		return d.detectHTTP(src)
	}

	return "", false, nil
}

func (d *S3Detector) detectHTTP(src string) (string, bool, error) {
	parts := strings.Split(src, "/")
	if len(parts) < 2 {
		return "", false, fmt.Errorf(
			"URL is not a valid S3 URL")
	}

	hostParts := strings.Split(parts[0], ".")
	if len(hostParts) == 3 {
		return d.detectPathStyle(hostParts[0], parts[1:])
	} else if len(hostParts) == 4 {
		return d.detectVhostStyle(hostParts[1], hostParts[0], parts[1:])
	} else {
		return "", false, fmt.Errorf(
			"URL is not a valid S3 URL")
	}
}

func (d *S3Detector) detectPathStyle(region string, parts []string) (string, bool, error) {
	urlStr := fmt.Sprintf("https://%s.amazonaws.com/%s", region, strings.Join(parts, "/"))
	url, err := url.Parse(urlStr)
	if err != nil {
		return "", false, fmt.Errorf("error parsing S3 URL: %s", err)
	}

	return "s3::" + url.String(), true, nil
}

func (d *S3Detector) detectVhostStyle(region, bucket string, parts []string) (string, bool, error) {
	urlStr := fmt.Sprintf("https://%s.amazonaws.com/%s/%s", region, bucket, strings.Join(parts, "/"))
	url, err := url.Parse(urlStr)
	if err != nil {
		return "", false, fmt.Errorf("error parsing S3 URL: %s", err)
	}

	return "s3::" + url.String(), true, nil
}
