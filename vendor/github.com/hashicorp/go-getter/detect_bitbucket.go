package getter

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// BitBucketDetector implements Detector to detect BitBucket URLs and turn
// them into URLs that the Git or Hg Getter can understand.
type BitBucketDetector struct{}

func (d *BitBucketDetector) Detect(src, _ string) (string, bool, error) {
	if len(src) == 0 {
		return "", false, nil
	}

	if strings.HasPrefix(src, "bitbucket.org/") {
		return d.detectHTTP(src)
	}

	return "", false, nil
}

func (d *BitBucketDetector) detectHTTP(src string) (string, bool, error) {
	u, err := url.Parse("https://" + src)
	if err != nil {
		return "", true, fmt.Errorf("error parsing BitBucket URL: %s", err)
	}

	// We need to get info on this BitBucket repository to determine whether
	// it is Git or Hg.
	var info struct {
		SCM string `json:"scm"`
	}
	infoUrl := "https://api.bitbucket.org/1.0/repositories" + u.Path
	resp, err := http.Get(infoUrl)
	if err != nil {
		return "", true, fmt.Errorf("error looking up BitBucket URL: %s", err)
	}
	if resp.StatusCode == 403 {
		// A private repo
		return "", true, fmt.Errorf(
			"shorthand BitBucket URL can't be used for private repos, " +
				"please use a full URL")
	}
	dec := json.NewDecoder(resp.Body)
	if err := dec.Decode(&info); err != nil {
		return "", true, fmt.Errorf("error looking up BitBucket URL: %s", err)
	}

	switch info.SCM {
	case "git":
		if !strings.HasSuffix(u.Path, ".git") {
			u.Path += ".git"
		}

		return "git::" + u.String(), true, nil
	case "hg":
		return "hg::" + u.String(), true, nil
	default:
		return "", true, fmt.Errorf("unknown BitBucket SCM type: %s", info.SCM)
	}
}
