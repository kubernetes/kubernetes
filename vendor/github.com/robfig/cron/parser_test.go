package cron

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestRange(t *testing.T) {
	zero := uint64(0)
	ranges := []struct {
		expr     string
		min, max uint
		expected uint64
		err      string
	}{
		{"5", 0, 7, 1 << 5, ""},
		{"0", 0, 7, 1 << 0, ""},
		{"7", 0, 7, 1 << 7, ""},

		{"5-5", 0, 7, 1 << 5, ""},
		{"5-6", 0, 7, 1<<5 | 1<<6, ""},
		{"5-7", 0, 7, 1<<5 | 1<<6 | 1<<7, ""},

		{"5-6/2", 0, 7, 1 << 5, ""},
		{"5-7/2", 0, 7, 1<<5 | 1<<7, ""},
		{"5-7/1", 0, 7, 1<<5 | 1<<6 | 1<<7, ""},

		{"*", 1, 3, 1<<1 | 1<<2 | 1<<3 | starBit, ""},
		{"*/2", 1, 3, 1<<1 | 1<<3 | starBit, ""},

		{"5--5", 0, 0, zero, "Too many hyphens"},
		{"jan-x", 0, 0, zero, "Failed to parse int from"},
		{"2-x", 1, 5, zero, "Failed to parse int from"},
		{"*/-12", 0, 0, zero, "Negative number"},
		{"*//2", 0, 0, zero, "Too many slashes"},
		{"1", 3, 5, zero, "below minimum"},
		{"6", 3, 5, zero, "above maximum"},
		{"5-3", 3, 5, zero, "beyond end of range"},
		{"*/0", 0, 0, zero, "should be a positive number"},
	}

	for _, c := range ranges {
		actual, err := getRange(c.expr, bounds{c.min, c.max, nil})
		if len(c.err) != 0 && (err == nil || !strings.Contains(err.Error(), c.err)) {
			t.Errorf("%s => expected %v, got %v", c.expr, c.err, err)
		}
		if len(c.err) == 0 && err != nil {
			t.Error("%s => unexpected error %v", c.expr, err)
		}
		if actual != c.expected {
			t.Errorf("%s => expected %d, got %d", c.expr, c.expected, actual)
		}
	}
}

func TestField(t *testing.T) {
	fields := []struct {
		expr     string
		min, max uint
		expected uint64
	}{
		{"5", 1, 7, 1 << 5},
		{"5,6", 1, 7, 1<<5 | 1<<6},
		{"5,6,7", 1, 7, 1<<5 | 1<<6 | 1<<7},
		{"1,5-7/2,3", 1, 7, 1<<1 | 1<<5 | 1<<7 | 1<<3},
	}

	for _, c := range fields {
		actual, _ := getField(c.expr, bounds{c.min, c.max, nil})
		if actual != c.expected {
			t.Errorf("%s => expected %d, got %d", c.expr, c.expected, actual)
		}
	}
}

func TestAll(t *testing.T) {
	allBits := []struct {
		r        bounds
		expected uint64
	}{
		{minutes, 0xfffffffffffffff}, // 0-59: 60 ones
		{hours, 0xffffff},            // 0-23: 24 ones
		{dom, 0xfffffffe},            // 1-31: 31 ones, 1 zero
		{months, 0x1ffe},             // 1-12: 12 ones, 1 zero
		{dow, 0x7f},                  // 0-6: 7 ones
	}

	for _, c := range allBits {
		actual := all(c.r) // all() adds the starBit, so compensate for that..
		if c.expected|starBit != actual {
			t.Errorf("%d-%d/%d => expected %b, got %b",
				c.r.min, c.r.max, 1, c.expected|starBit, actual)
		}
	}
}

func TestBits(t *testing.T) {
	bits := []struct {
		min, max, step uint
		expected       uint64
	}{
		{0, 0, 1, 0x1},
		{1, 1, 1, 0x2},
		{1, 5, 2, 0x2a}, // 101010
		{1, 4, 2, 0xa},  // 1010
	}

	for _, c := range bits {
		actual := getBits(c.min, c.max, c.step)
		if c.expected != actual {
			t.Errorf("%d-%d/%d => expected %b, got %b",
				c.min, c.max, c.step, c.expected, actual)
		}
	}
}

func TestParse(t *testing.T) {
	entries := []struct {
		expr     string
		expected Schedule
		err      string
	}{
		{
			expr: "* 5 * * * *",
			expected: &SpecSchedule{
				Second: all(seconds),
				Minute: 1 << 5,
				Hour:   all(hours),
				Dom:    all(dom),
				Month:  all(months),
				Dow:    all(dow),
			},
		},
		{
			expr: "* 5 j * * *",
			err:  "Failed to parse int from",
		},
		{
			expr:     "@every 5m",
			expected: ConstantDelaySchedule{Delay: time.Duration(5) * time.Minute},
		},
		{
			expr: "@every Xm",
			err:  "Failed to parse duration",
		},
		{
			expr: "@yearly",
			expected: &SpecSchedule{
				Second: 1 << seconds.min,
				Minute: 1 << minutes.min,
				Hour:   1 << hours.min,
				Dom:    1 << dom.min,
				Month:  1 << months.min,
				Dow:    all(dow),
			},
		},
		{
			expr: "@annually",
			expected: &SpecSchedule{
				Second: 1 << seconds.min,
				Minute: 1 << minutes.min,
				Hour:   1 << hours.min,
				Dom:    1 << dom.min,
				Month:  1 << months.min,
				Dow:    all(dow),
			},
		},
		{
			expr: "@unrecognized",
			err:  "Unrecognized descriptor",
		},
		{
			expr: "* * * *",
			err:  "Expected 5 to 6 fields",
		},
		{
			expr: "",
			err:  "Empty spec string",
		},
	}

	for _, c := range entries {
		actual, err := Parse(c.expr)
		if len(c.err) != 0 && (err == nil || !strings.Contains(err.Error(), c.err)) {
			t.Errorf("%s => expected %v, got %v", c.expr, c.err, err)
		}
		if len(c.err) == 0 && err != nil {
			t.Errorf("%s => unexpected error %v", c.expr, err)
		}
		if !reflect.DeepEqual(actual, c.expected) {
			t.Errorf("%s => expected %b, got %b", c.expr, c.expected, actual)
		}
	}
}

func TestStandardSpecSchedule(t *testing.T) {
	entries := []struct {
		expr     string
		expected Schedule
		err      string
	}{
		{
			expr:     "5 * * * *",
			expected: &SpecSchedule{1 << seconds.min, 1 << 5, all(hours), all(dom), all(months), all(dow)},
		},
		{
			expr:     "@every 5m",
			expected: ConstantDelaySchedule{time.Duration(5) * time.Minute},
		},
		{
			expr: "5 j * * *",
			err:  "Failed to parse int from",
		},
		{
			expr: "* * * *",
			err:  "Expected exactly 5 fields",
		},
	}

	for _, c := range entries {
		actual, err := ParseStandard(c.expr)
		if len(c.err) != 0 && (err == nil || !strings.Contains(err.Error(), c.err)) {
			t.Errorf("%s => expected %v, got %v", c.expr, c.err, err)
		}
		if len(c.err) == 0 && err != nil {
			t.Errorf("%s => unexpected error %v", c.expr, err)
		}
		if !reflect.DeepEqual(actual, c.expected) {
			t.Errorf("%s => expected %b, got %b", c.expr, c.expected, actual)
		}
	}
}
