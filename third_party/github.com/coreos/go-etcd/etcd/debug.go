package etcd

import (
	"io/ioutil"
	"log"
	"strings"
)

var logger *etcdLogger

func SetLogger(l *log.Logger) {
	logger = &etcdLogger{l}
}

func GetLogger() *log.Logger {
	return logger.log
}

type etcdLogger struct {
	log *log.Logger
}

func (p *etcdLogger) Debug(args ...interface{}) {
	args[0] = "DEBUG: " + args[0].(string)
	p.log.Println(args)
}

func (p *etcdLogger) Debugf(fmt string, args ...interface{}) {
	args[0] = "DEBUG: " + args[0].(string)
	// Append newline if necessary
	if !strings.HasSuffix(fmt, "\n") {
		fmt = fmt + "\n"
	}
	p.log.Printf(fmt, args)
}

func (p *etcdLogger) Warning(args ...interface{}) {
	args[0] = "WARNING: " + args[0].(string)
	p.log.Println(args)
}

func (p *etcdLogger) Warningf(fmt string, args ...interface{}) {
	// Append newline if necessary
	if !strings.HasSuffix(fmt, "\n") {
		fmt = fmt + "\n"
	}
	args[0] = "WARNING: " + args[0].(string)
	p.log.Printf(fmt, args)
}

func init() {
	// Default logger uses the go default log.
	SetLogger(log.New(ioutil.Discard, "go-etcd", log.LstdFlags))
}
