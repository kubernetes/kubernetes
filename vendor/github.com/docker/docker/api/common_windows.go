package api // import "github.com/docker/docker/api"

// MinVersion represents Minimum REST API version supported
// Technically the first daemon API version released on Windows is v1.25 in
// engine version 1.13. However, some clients are explicitly using downlevel
// APIs (e.g. docker-compose v2.1 file format) and that is just too restrictive.
// Hence also allowing 1.24 on Windows.
const MinVersion string = "1.24"
