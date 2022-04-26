package commands

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/golangci/golangci-lint/internal/cache"
	"github.com/golangci/golangci-lint/pkg/exitcodes"
	"github.com/golangci/golangci-lint/pkg/fsutils"
	"github.com/golangci/golangci-lint/pkg/logutils"
)

func (e *Executor) initCache() {
	cacheCmd := &cobra.Command{
		Use:   "cache",
		Short: "Cache control and information",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) != 0 {
				e.log.Fatalf("Usage: golangci-lint cache")
			}
			if err := cmd.Help(); err != nil {
				e.log.Fatalf("Can't run cache: %s", err)
			}
		},
	}
	e.rootCmd.AddCommand(cacheCmd)

	cacheCmd.AddCommand(&cobra.Command{
		Use:   "clean",
		Short: "Clean cache",
		Run:   e.executeCleanCache,
	})
	cacheCmd.AddCommand(&cobra.Command{
		Use:   "status",
		Short: "Show cache status",
		Run:   e.executeCacheStatus,
	})

	// TODO: add trim command?
}

func (e *Executor) executeCleanCache(_ *cobra.Command, args []string) {
	if len(args) != 0 {
		e.log.Fatalf("Usage: golangci-lint cache clean")
	}

	cacheDir := cache.DefaultDir()
	if err := os.RemoveAll(cacheDir); err != nil {
		e.log.Fatalf("Failed to remove dir %s: %s", cacheDir, err)
	}

	os.Exit(exitcodes.Success)
}

func (e *Executor) executeCacheStatus(_ *cobra.Command, args []string) {
	if len(args) != 0 {
		e.log.Fatalf("Usage: golangci-lint cache status")
	}

	cacheDir := cache.DefaultDir()
	fmt.Fprintf(logutils.StdOut, "Dir: %s\n", cacheDir)
	cacheSizeBytes, err := dirSizeBytes(cacheDir)
	if err == nil {
		fmt.Fprintf(logutils.StdOut, "Size: %s\n", fsutils.PrettifyBytesCount(cacheSizeBytes))
	}

	os.Exit(exitcodes.Success)
}

func dirSizeBytes(path string) (int64, error) {
	var size int64
	err := filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return err
	})
	return size, err
}
