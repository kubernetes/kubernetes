package context

import (
	"fmt"
	"time"

	log "github.com/Sirupsen/logrus"

	"github.com/emccode/libstorage/api/types"
)

const (
	keyFieldOffset = 1000
)

func (ctx *lsc) ctxFields() map[string]interface{} {

	fields := map[string]interface{}{
		"time": time.Now().UTC().Unix(),
	}

	for key := Key(keyLoggable - 1); key > keyEOF; key-- {
		ctxFieldsProcessValue(key.String(), ctx.Value(key), fields)
	}

	for key := range CustomLoggerKeys() {
		ctxFieldsProcessValue(key, ctx.Value(key), fields)
	}

	return fields
}

func ctxFieldsProcessValue(
	key interface{}, val interface{}, fields map[string]interface{}) {

	var keyName string
	switch tk := key.(type) {
	case string:
		keyName = tk
	case fmt.Stringer:
		keyName = tk.String()
	default:
		keyName = fmt.Sprintf("%v", key)
	}

	switch tv := val.(type) {
	case types.Client:
		if tv.OS() != nil {
			fields["osDriver"] = tv.OS().Name()
		}
		if tv.Storage() != nil {
			fields["storageDriver"] = tv.Storage().Name()
		}
		if tv.Integration() != nil {
			fields["integrationDriver"] = tv.Integration().Name()
		}
	case types.ContextLoggerFieldAware:
		k, v := tv.ContextLoggerField()
		fields[k] = v
	case types.ContextLoggerFieldsAware:
		for k, v := range tv.ContextLoggerFields() {
			fields[k] = v
		}
	case string:
		fields[keyName] = tv
	case hasName:
		fields[keyName] = tv.Name()
	case hasID:
		fields[keyName] = tv.ID()
	case fmt.Stringer:
		fields[keyName] = tv.String()
	case bool, uint, uint32, uint64, int, int32, int64, float32, float64:
		fields[keyName] = fmt.Sprintf("%v", tv)
	}
}

func (ctx *lsc) addkeyFieldOffsetFieldsToEntry(entry *log.Entry) {
	fields := ctx.ctxFields()
	for k, v := range fields {
		entry.Data[k] = v
	}
}

func (ctx *lsc) logger() log.FieldLogger {
	return ctx.Value(LoggerKey).(log.FieldLogger)
}

func (ctx *lsc) WithField(key string, value interface{}) *log.Entry {
	entry := ctx.logger().WithField(key, value)
	ctx.addkeyFieldOffsetFieldsToEntry(entry)
	return entry
}
func (ctx *lsc) WithFields(fields log.Fields) *log.Entry {
	entry := ctx.logger().WithFields(fields)
	ctx.addkeyFieldOffsetFieldsToEntry(entry)
	return entry
}
func (ctx *lsc) WithError(err error) *log.Entry {
	entry := ctx.logger().WithError(err)
	ctx.addkeyFieldOffsetFieldsToEntry(entry)
	return entry
}

func (ctx *lsc) Debugf(format string, args ...interface{}) {
	ctx.Debug(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Infof(format string, args ...interface{}) {
	ctx.Info(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Printf(format string, args ...interface{}) {
	ctx.Print(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Warnf(format string, args ...interface{}) {
	ctx.Warn(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Warningf(format string, args ...interface{}) {
	ctx.Warning(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Errorf(format string, args ...interface{}) {
	ctx.Error(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Fatalf(format string, args ...interface{}) {
	ctx.Fatal(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Panicf(format string, args ...interface{}) {
	ctx.Panic(fmt.Sprintf(format, args...))
}

func (ctx *lsc) Debug(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Debug(args...)
}

func (ctx *lsc) Info(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Info(args...)
}

func (ctx *lsc) Print(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Print(args...)
}

func (ctx *lsc) Warn(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Warn(args...)
}

func (ctx *lsc) Warning(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Warning(args...)
}

func (ctx *lsc) Error(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Error(args...)
}

func (ctx *lsc) Fatal(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Fatal(args...)
}

func (ctx *lsc) Panic(args ...interface{}) {
	ctx.logger().WithFields(ctx.ctxFields()).Panic(args...)
}

func (ctx *lsc) Debugln(args ...interface{}) {
	ctx.logger().Debug(args...)
}

func (ctx *lsc) Infoln(args ...interface{}) {
	ctx.logger().Info(args...)
}

func (ctx *lsc) Println(args ...interface{}) {
	ctx.logger().Print(args...)
}

func (ctx *lsc) Warnln(args ...interface{}) {
	ctx.logger().Warn(args...)
}

func (ctx *lsc) Warningln(args ...interface{}) {
	ctx.logger().Warning(args...)
}

func (ctx *lsc) Errorln(args ...interface{}) {
	ctx.logger().Error(args...)
}

func (ctx *lsc) Fatalln(args ...interface{}) {
	ctx.logger().Fatal(args...)
}

func (ctx *lsc) Panicln(args ...interface{}) {
	ctx.logger().Panic(args...)
}
