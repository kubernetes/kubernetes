package kubelet

import (
	"net/url"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
)

func Test_journalArgs_Args(t *testing.T) {
	tests := []struct {
		name string
		args journalArgs
		want []string
	}{
		{args: journalArgs{}, want: []string{"--utc", "--no-pager"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.args.Args(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("journalArgs.Args() = %v, want %v", got, tt.want)
			}
		})
	}
}

func repeatString(s string, times int) []string {
	var arr []string
	for i := 0; i < times; i++ {
		arr = append(arr, s)
	}
	return arr
}

func Test_newJournalArgsFromURL(t *testing.T) {
	tests := []struct {
		name    string
		query   url.Values
		want    *journalArgs
		wantErr bool
	}{
		{query: url.Values{}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true}},
		{query: url.Values{"unknown": []string{"true"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true}},

		{query: url.Values{"since": []string{""}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true}},
		{query: url.Values{"since": []string{"1m"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Since: "1m"}},
		{query: url.Values{"since": []string{"12d"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Since: "12d"}},
		{query: url.Values{"since": []string{"516s"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Since: "516s"}},
		{query: url.Values{"since": []string{"-516s"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Since: "-516s"}},
		{query: url.Values{"since": []string{"1y"}}, wantErr: true},
		{query: url.Values{"since": []string{"1"}}, wantErr: true},
		{query: url.Values{"since": []string{"y"}}, wantErr: true},
		{query: url.Values{"since": []string{"-1"}}, wantErr: true},
		{query: url.Values{"since": []string{"-y"}}, wantErr: true},
		{query: url.Values{"since": []string{"2019-12-04 02:00:00"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Since: "2019-12-04 02:00:00"}},
		{query: url.Values{"since": []string{"2019-12-04 02:00:00.000"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Since: "2019-12-04 02:00:00.000"}},
		{query: url.Values{"since": []string{"2019-12-04 02"}}, wantErr: true},
		{query: url.Values{"since": []string{"2019-12-04 02:00"}}, wantErr: true},
		{query: url.Values{"since": []string{"2019-12-04T02:00:00Z"}}, wantErr: true},

		{query: url.Values{"until": []string{""}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true}},
		{query: url.Values{"until": []string{"1m"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Until: "1m"}},
		{query: url.Values{"until": []string{"-y"}}, wantErr: true},
		{query: url.Values{"until": []string{"2019-12-04 02:00:00"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Until: "2019-12-04 02:00:00"}},
		{query: url.Values{"until": []string{"2019-12-04 02:00:00.000"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Until: "2019-12-04 02:00:00.000"}},
		{query: url.Values{"until": []string{"2019-12-04 02"}}, wantErr: true},
		{query: url.Values{"until": []string{"2019-12-04 02:00"}}, wantErr: true},
		{query: url.Values{"until": []string{"2019-12-04T02:00:00Z"}}, wantErr: true},

		{query: url.Values{"output": []string{"short", "precise"}}, want: &journalArgs{Timeout: 30, CaseSensitive: true, Format: "short"}},
		{query: url.Values{"output": []string{"short"}}, want: &journalArgs{Timeout: 30, CaseSensitive: true, Format: "short"}},
		{query: url.Values{"output": []string{""}}, want: &journalArgs{Timeout: 30, CaseSensitive: true, Format: "short-precise"}},

		{query: url.Values{"tail": []string{"100"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Tail: 100}},
		{query: url.Values{"tail": []string{"10000000"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Tail: 100000}},

		{query: url.Values{"case-sensitive": []string{"false"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: false}},
		{query: url.Values{"case-sensitive": []string{"0"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: false}},
		{query: url.Values{"case-sensitive": []string{"a"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: false}},

		{query: url.Values{"grep": []string{"string"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Pattern: "string"}},
		{name: "long grep", query: url.Values{"grep": []string{strings.Repeat("abc", 100)}}, wantErr: true},
		{name: "total grep", query: url.Values{"grep": repeatString(strings.Repeat("a", 100), 2)}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Pattern: strings.Repeat("a", 100)}},

		{query: url.Values{"unit": []string{"a"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Units: []string{"a"}}},
		{query: url.Values{"unit": []string{""}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Units: []string{""}}},
		{query: url.Values{"unit": []string{"a", "b"}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true, Units: []string{"a", "b"}}},

		{name: "long unit", query: url.Values{"unit": []string{strings.Repeat("abc", 100)}}, wantErr: true},
		{name: "total unit", query: url.Values{"unit": repeatString(strings.Repeat("a", 100), 11)}, wantErr: true},

		{query: url.Values{"boot": []string{"0"}}, want: &journalArgs{Boot: intPtr(0), Timeout: 30, Format: "short-precise", CaseSensitive: true}},
		{query: url.Values{"boot": []string{"-23"}}, want: &journalArgs{Boot: intPtr(-23), Timeout: 30, Format: "short-precise", CaseSensitive: true}},
		{query: url.Values{"boot": []string{"-100"}}, want: &journalArgs{Boot: intPtr(-100), Timeout: 30, Format: "short-precise", CaseSensitive: true}},
		{name: "empty value", query: url.Values{"boot": []string{""}}, want: &journalArgs{Timeout: 30, Format: "short-precise", CaseSensitive: true}},
		{name: "boot below range", query: url.Values{"boot": []string{"-101"}}, wantErr: true},
		{name: "boot above range", query: url.Values{"boot": []string{"1"}}, wantErr: true},
		{name: "boot below range", query: url.Values{"boot": []string{"-9999"}}, wantErr: true},
		{name: "boot above range", query: url.Values{"boot": []string{"9999"}}, wantErr: true},
		{name: "boot not int", query: url.Values{"boot": []string{"a"}}, wantErr: true},
		{name: "boot", query: url.Values{"boot": []string{"x"}}, wantErr: true},
		{name: "boot", query: url.Values{"boot": []string{" "}}, wantErr: true},
	}
	for _, tt := range tests {
		name := tt.name
		if len(name) == 0 {
			name = tt.query.Encode()
		}
		t.Run(name, func(t *testing.T) {
			got, err := newJournalArgsFromURL(tt.query)
			if (err != nil) != tt.wantErr {
				t.Errorf("newJournalArgsFromURL() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("different: %s", diff.ObjectReflectDiff(tt.want, got))
			}
		})
	}
}

func intPtr(i int) *int {
	return &i
}
