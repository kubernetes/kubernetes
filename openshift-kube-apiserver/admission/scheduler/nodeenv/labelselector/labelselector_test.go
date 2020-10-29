package labelselector

import (
	"testing"
)

func TestLabelSelectorParse(t *testing.T) {
	tests := []struct {
		selector string
		labels   map[string]string
		valid    bool
	}{
		{
			selector: "",
			labels:   map[string]string{},
			valid:    true,
		},
		{
			selector: "   ",
			labels:   map[string]string{},
			valid:    true,
		},
		{
			selector: "x=a",
			labels:   map[string]string{"x": "a"},
			valid:    true,
		},
		{
			selector: "x=a,y=b,z=c",
			labels:   map[string]string{"x": "a", "y": "b", "z": "c"},
			valid:    true,
		},
		{
			selector: "x = a, y=b ,z  = c  ",
			labels:   map[string]string{"x": "a", "y": "b", "z": "c"},
			valid:    true,
		},
		{
			selector: "color=green, env = test ,service= front ",
			labels:   map[string]string{"color": "green", "env": "test", "service": "front"},
			valid:    true,
		},
		{
			selector: ",",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x,y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x=$y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x!=y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x==y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x=a||y=b",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x in (y)",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x notin (y)",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "node-role.kubernetes.io/infra=",
			labels:   map[string]string{"node-role.kubernetes.io/infra": ""},
			valid:    true,
		},
	}
	for _, test := range tests {
		labels, err := Parse(test.selector)
		if test.valid && err != nil {
			t.Errorf("selector: %s, expected no error but got: %s", test.selector, err)
		} else if !test.valid && err == nil {
			t.Errorf("selector: %s, expected an error", test.selector)
		}

		if !Equals(labels, test.labels) {
			t.Errorf("expected: %s but got: %s", test.labels, labels)
		}
	}
}

func TestLabelConflict(t *testing.T) {
	tests := []struct {
		labels1  map[string]string
		labels2  map[string]string
		conflict bool
	}{
		{
			labels1:  map[string]string{},
			labels2:  map[string]string{},
			conflict: false,
		},
		{
			labels1:  map[string]string{"env": "test"},
			labels2:  map[string]string{"infra": "true"},
			conflict: false,
		},
		{
			labels1:  map[string]string{"env": "test"},
			labels2:  map[string]string{"infra": "true", "env": "test"},
			conflict: false,
		},
		{
			labels1:  map[string]string{"env": "test"},
			labels2:  map[string]string{"env": "dev"},
			conflict: true,
		},
		{
			labels1:  map[string]string{"env": "test", "infra": "false"},
			labels2:  map[string]string{"infra": "true", "color": "blue"},
			conflict: true,
		},
	}
	for _, test := range tests {
		conflict := Conflicts(test.labels1, test.labels2)
		if conflict != test.conflict {
			t.Errorf("expected: %v but got: %v", test.conflict, conflict)
		}
	}
}

func TestLabelMerge(t *testing.T) {
	tests := []struct {
		labels1      map[string]string
		labels2      map[string]string
		mergedLabels map[string]string
	}{
		{
			labels1:      map[string]string{},
			labels2:      map[string]string{},
			mergedLabels: map[string]string{},
		},
		{
			labels1:      map[string]string{"infra": "true"},
			labels2:      map[string]string{},
			mergedLabels: map[string]string{"infra": "true"},
		},
		{
			labels1:      map[string]string{"infra": "true"},
			labels2:      map[string]string{"env": "test", "color": "blue"},
			mergedLabels: map[string]string{"infra": "true", "env": "test", "color": "blue"},
		},
	}
	for _, test := range tests {
		mergedLabels := Merge(test.labels1, test.labels2)
		if !Equals(mergedLabels, test.mergedLabels) {
			t.Errorf("expected: %v but got: %v", test.mergedLabels, mergedLabels)
		}
	}
}
