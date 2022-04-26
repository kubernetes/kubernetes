package fsutils

import (
	"fmt"
	"os"
	"sync"

	"github.com/pkg/errors"

	"github.com/golangci/golangci-lint/pkg/logutils"
)

type FileCache struct {
	files sync.Map
}

func NewFileCache() *FileCache {
	return &FileCache{}
}

func (fc *FileCache) GetFileBytes(filePath string) ([]byte, error) {
	cachedBytes, ok := fc.files.Load(filePath)
	if ok {
		return cachedBytes.([]byte), nil
	}

	fileBytes, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "can't read file %s", filePath)
	}

	fc.files.Store(filePath, fileBytes)
	return fileBytes, nil
}

func PrettifyBytesCount(n int64) string {
	const (
		Multiplexer = 1024
		KiB         = 1 * Multiplexer
		MiB         = KiB * Multiplexer
		GiB         = MiB * Multiplexer
	)

	if n >= GiB {
		return fmt.Sprintf("%.1fGiB", float64(n)/GiB)
	}
	if n >= MiB {
		return fmt.Sprintf("%.1fMiB", float64(n)/MiB)
	}
	if n >= KiB {
		return fmt.Sprintf("%.1fKiB", float64(n)/KiB)
	}
	return fmt.Sprintf("%dB", n)
}

func (fc *FileCache) PrintStats(log logutils.Log) {
	var size int64
	var mapLen int
	fc.files.Range(func(_, fileBytes interface{}) bool {
		mapLen++
		size += int64(len(fileBytes.([]byte)))

		return true
	})

	log.Infof("File cache stats: %d entries of total size %s", mapLen, PrettifyBytesCount(size))
}
