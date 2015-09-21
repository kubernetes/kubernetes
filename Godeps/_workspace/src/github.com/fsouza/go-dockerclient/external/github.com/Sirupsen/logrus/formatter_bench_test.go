package logrus

import (
	"fmt"
	"testing"
	"time"
)

// smallFields is a small size data set for benchmarking
var smallFields = Fields{
	"foo":   "bar",
	"baz":   "qux",
	"one":   "two",
	"three": "four",
}

// largeFields is a large size data set for benchmarking
var largeFields = Fields{
	"foo":       "bar",
	"baz":       "qux",
	"one":       "two",
	"three":     "four",
	"five":      "six",
	"seven":     "eight",
	"nine":      "ten",
	"eleven":    "twelve",
	"thirteen":  "fourteen",
	"fifteen":   "sixteen",
	"seventeen": "eighteen",
	"nineteen":  "twenty",
	"a":         "b",
	"c":         "d",
	"e":         "f",
	"g":         "h",
	"i":         "j",
	"k":         "l",
	"m":         "n",
	"o":         "p",
	"q":         "r",
	"s":         "t",
	"u":         "v",
	"w":         "x",
	"y":         "z",
	"this":      "will",
	"make":      "thirty",
	"entries":   "yeah",
}

var errorFields = Fields{
	"foo": fmt.Errorf("bar"),
	"baz": fmt.Errorf("qux"),
}

func BenchmarkErrorTextFormatter(b *testing.B) {
	doBenchmark(b, &TextFormatter{DisableColors: true}, errorFields)
}

func BenchmarkSmallTextFormatter(b *testing.B) {
	doBenchmark(b, &TextFormatter{DisableColors: true}, smallFields)
}

func BenchmarkLargeTextFormatter(b *testing.B) {
	doBenchmark(b, &TextFormatter{DisableColors: true}, largeFields)
}

func BenchmarkSmallColoredTextFormatter(b *testing.B) {
	doBenchmark(b, &TextFormatter{ForceColors: true}, smallFields)
}

func BenchmarkLargeColoredTextFormatter(b *testing.B) {
	doBenchmark(b, &TextFormatter{ForceColors: true}, largeFields)
}

func BenchmarkSmallJSONFormatter(b *testing.B) {
	doBenchmark(b, &JSONFormatter{}, smallFields)
}

func BenchmarkLargeJSONFormatter(b *testing.B) {
	doBenchmark(b, &JSONFormatter{}, largeFields)
}

func doBenchmark(b *testing.B, formatter Formatter, fields Fields) {
	entry := &Entry{
		Time:    time.Time{},
		Level:   InfoLevel,
		Message: "message",
		Data:    fields,
	}
	var d []byte
	var err error
	for i := 0; i < b.N; i++ {
		d, err = formatter.Format(entry)
		if err != nil {
			b.Fatal(err)
		}
		b.SetBytes(int64(len(d)))
	}
}
