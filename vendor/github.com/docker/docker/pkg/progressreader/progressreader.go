package progressreader

import (
	"io"

	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/streamformatter"
)

// Reader with progress bar
type Config struct {
	In         io.ReadCloser // Stream to read from
	Out        io.Writer     // Where to send progress bar to
	Formatter  *streamformatter.StreamFormatter
	Size       int
	Current    int
	LastUpdate int
	NewLines   bool
	ID         string
	Action     string
}

func New(newReader Config) *Config {
	return &newReader
}

func (config *Config) Read(p []byte) (n int, err error) {
	read, err := config.In.Read(p)
	config.Current += read
	updateEvery := 1024 * 512 //512kB
	if config.Size > 0 {
		// Update progress for every 1% read if 1% < 512kB
		if increment := int(0.01 * float64(config.Size)); increment < updateEvery {
			updateEvery = increment
		}
	}
	if config.Current-config.LastUpdate > updateEvery || err != nil {
		updateProgress(config)
		config.LastUpdate = config.Current
	}

	if err != nil && read == 0 {
		updateProgress(config)
		if config.NewLines {
			config.Out.Write(config.Formatter.FormatStatus("", ""))
		}
	}
	return read, err
}

func (config *Config) Close() error {
	if config.Current < config.Size {
		//print a full progress bar when closing prematurely
		config.Current = config.Size
		updateProgress(config)
	}
	return config.In.Close()
}

func updateProgress(config *Config) {
	progress := jsonmessage.JSONProgress{Current: config.Current, Total: config.Size}
	fmtMessage := config.Formatter.FormatProgress(config.ID, config.Action, &progress)
	config.Out.Write(fmtMessage)
}
