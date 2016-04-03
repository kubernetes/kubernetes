package jsonfilelog

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/pkg/jsonlog"
	"github.com/docker/docker/pkg/timeutils"
	"github.com/docker/docker/pkg/units"
)

const (
	Name = "json-file"
)

// JSONFileLogger is Logger implementation for default docker logging:
// JSON objects to file
type JSONFileLogger struct {
	buf      *bytes.Buffer
	f        *os.File   // store for closing
	mu       sync.Mutex // protects buffer
	capacity int64      //maximum size of each file
	n        int        //maximum number of files
	ctx      logger.Context
}

func init() {
	if err := logger.RegisterLogDriver(Name, New); err != nil {
		logrus.Fatal(err)
	}
	if err := logger.RegisterLogOptValidator(Name, ValidateLogOpt); err != nil {
		logrus.Fatal(err)
	}
}

// New creates new JSONFileLogger which writes to filename
func New(ctx logger.Context) (logger.Logger, error) {
	log, err := os.OpenFile(ctx.LogPath, os.O_RDWR|os.O_APPEND|os.O_CREATE, 0600)
	if err != nil {
		return nil, err
	}
	var capval int64 = -1
	if capacity, ok := ctx.Config["max-size"]; ok {
		var err error
		capval, err = units.FromHumanSize(capacity)
		if err != nil {
			return nil, err
		}
	}
	var maxFiles int = 1
	if maxFileString, ok := ctx.Config["max-file"]; ok {
		maxFiles, err = strconv.Atoi(maxFileString)
		if err != nil {
			return nil, err
		}
		if maxFiles < 1 {
			return nil, fmt.Errorf("max-files cannot be less than 1.")
		}
	}
	return &JSONFileLogger{
		f:        log,
		buf:      bytes.NewBuffer(nil),
		ctx:      ctx,
		capacity: capval,
		n:        maxFiles,
	}, nil
}

// Log converts logger.Message to jsonlog.JSONLog and serializes it to file
func (l *JSONFileLogger) Log(msg *logger.Message) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	timestamp, err := timeutils.FastMarshalJSON(msg.Timestamp)
	if err != nil {
		return err
	}
	err = (&jsonlog.JSONLogBytes{Log: append(msg.Line, '\n'), Stream: msg.Source, Created: timestamp}).MarshalJSONBuf(l.buf)
	if err != nil {
		return err
	}
	l.buf.WriteByte('\n')
	_, err = writeLog(l)
	return err
}

func writeLog(l *JSONFileLogger) (int64, error) {
	if l.capacity == -1 {
		return writeToBuf(l)
	}
	meta, err := l.f.Stat()
	if err != nil {
		return -1, err
	}
	if meta.Size() >= l.capacity {
		name := l.f.Name()
		if err := l.f.Close(); err != nil {
			return -1, err
		}
		if err := rotate(name, l.n); err != nil {
			return -1, err
		}
		file, err := os.OpenFile(name, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
		if err != nil {
			return -1, err
		}
		l.f = file
	}
	return writeToBuf(l)
}

func writeToBuf(l *JSONFileLogger) (int64, error) {
	i, err := l.buf.WriteTo(l.f)
	if err != nil {
		l.buf = bytes.NewBuffer(nil)
	}
	return i, err
}

func rotate(name string, n int) error {
	if n < 2 {
		return nil
	}
	for i := n - 1; i > 1; i-- {
		oldFile := name + "." + strconv.Itoa(i)
		replacingFile := name + "." + strconv.Itoa(i-1)
		if err := backup(oldFile, replacingFile); err != nil {
			return err
		}
	}
	if err := backup(name+".1", name); err != nil {
		return err
	}
	return nil
}

func backup(old, curr string) error {
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		err := os.Remove(old)
		if err != nil {
			return err
		}
	}
	if _, err := os.Stat(curr); os.IsNotExist(err) {
		if f, err := os.Create(curr); err != nil {
			return err
		} else {
			f.Close()
		}
	}
	return os.Rename(curr, old)
}

func ValidateLogOpt(cfg map[string]string) error {
	for key := range cfg {
		switch key {
		case "max-file":
		case "max-size":
		default:
			return fmt.Errorf("unknown log opt '%s' for json-file log driver", key)
		}
	}
	return nil
}

func (l *JSONFileLogger) ReadLog(args ...string) (io.Reader, error) {
	pth := l.ctx.LogPath
	if len(args) > 0 {
		//check if args[0] is an integer index
		index, err := strconv.ParseInt(args[0], 0, 0)
		if err != nil {
			return nil, err
		}
		if index > 0 {
			pth = pth + "." + args[0]
		}
	}
	return os.Open(pth)
}

func (l *JSONFileLogger) LogPath() string {
	return l.ctx.LogPath
}

// Close closes underlying file
func (l *JSONFileLogger) Close() error {
	return l.f.Close()
}

// Name returns name of this logger
func (l *JSONFileLogger) Name() string {
	return Name
}
