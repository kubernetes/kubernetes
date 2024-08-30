package log

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"reflect"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// TimeFormat is [time.RFC3339Nano] with nanoseconds padded using
// zeros to ensure the formatted time is always the same number of
// characters.
// Based on RFC3339NanoFixed from github.com/containerd/log
const TimeFormat = "2006-01-02T15:04:05.000000000Z07:00"

func FormatTime(t time.Time) string {
	return t.Format(TimeFormat)
}

// DurationFormat formats a [time.Duration] log entry.
//
// A nil value signals an error with the formatting.
type DurationFormat func(time.Duration) interface{}

func DurationFormatString(d time.Duration) interface{}       { return d.String() }
func DurationFormatSeconds(d time.Duration) interface{}      { return d.Seconds() }
func DurationFormatMilliseconds(d time.Duration) interface{} { return d.Milliseconds() }

// FormatIO formats net.Conn and other types that have an `Addr()` or `Name()`.
//
// See FormatEnabled for more information.
func FormatIO(ctx context.Context, v interface{}) string {
	m := make(map[string]string)
	m["type"] = reflect.TypeOf(v).String()

	switch t := v.(type) {
	case net.Conn:
		m["localAddress"] = formatAddr(t.LocalAddr())
		m["remoteAddress"] = formatAddr(t.RemoteAddr())
	case interface{ Addr() net.Addr }:
		m["address"] = formatAddr(t.Addr())
	default:
		return Format(ctx, t)
	}

	return Format(ctx, m)
}

func formatAddr(a net.Addr) string {
	return a.Network() + "://" + a.String()
}

// Format formats an object into a JSON string, without any indendtation or
// HTML escapes.
// Context is used to output a log waring if the conversion fails.
//
// This is intended primarily for `trace.StringAttribute()`
func Format(ctx context.Context, v interface{}) string {
	b, err := encode(v)
	if err != nil {
		// logging errors aren't really warning worthy, and can potentially spam a lot of logs out
		G(ctx).WithFields(logrus.Fields{
			logrus.ErrorKey: err,
			"type":          fmt.Sprintf("%T", v),
		}).Debug("could not format value")
		return ""
	}

	return string(b)
}

func encode(v interface{}) (_ []byte, err error) {
	if m, ok := v.(proto.Message); ok {
		// use canonical JSON encoding for protobufs (instead of [encoding/json])
		// https://protobuf.dev/programming-guides/proto3/#json
		var b []byte
		b, err = protojson.MarshalOptions{
			AllowPartial: true,
			// protobuf defaults to camel case for JSON encoding; use proto field name instead (snake case)
			UseProtoNames: true,
		}.Marshal(m)
		if err == nil {
			// the protojson marshaller tries to unmarshal anypb.Any fields, which can
			// fail for types encoded with "github.com/containerd/typeurl/v2"
			// we can try creating a dedicated protoregistry.MessageTypeResolver that uses typeurl, but, its
			// more robust to fall back on json marshalling for errors in general
			return b, nil
		}

	}

	buf := &bytes.Buffer{}
	enc := json.NewEncoder(buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "")

	if jErr := enc.Encode(v); jErr != nil {
		if err != nil {
			return nil, fmt.Errorf("protojson encoding: %w; json encoding: %w", err, jErr)
		}
		return nil, fmt.Errorf("json encoding: %w", jErr)
	}

	// encoder.Encode appends a newline to the end
	return bytes.TrimSpace(buf.Bytes()), nil
}
