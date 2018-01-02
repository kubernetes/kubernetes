package local

import (
	"os"
	"time"
)

func getStartTime(fi os.FileInfo) time.Time {
	return fi.ModTime()
}

func getATime(fi os.FileInfo) time.Time {
	return fi.ModTime()
}
