package typed_test

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

func TestComparisonExcludeFields(t *testing.T) {
	cases := []struct {
		name       string
		Comparison *typed.Comparison
		Remove     *fieldpath.Set
		Expect     *typed.Comparison
		Fails      bool
	}{
		{
			name: "works on nil set",
			Comparison: &typed.Comparison{
				Added:    fieldpath.NewSet(fieldpath.MakePathOrDie("a")),
				Modified: fieldpath.NewSet(fieldpath.MakePathOrDie("b")),
				Removed:  fieldpath.NewSet(fieldpath.MakePathOrDie("c")),
			},
			Remove: nil,
			Expect: &typed.Comparison{
				Added:    fieldpath.NewSet(fieldpath.MakePathOrDie("a")),
				Modified: fieldpath.NewSet(fieldpath.MakePathOrDie("b")),
				Removed:  fieldpath.NewSet(fieldpath.MakePathOrDie("c")),
			},
		},
		{
			name: "works on empty set",
			Comparison: &typed.Comparison{
				Added:    fieldpath.NewSet(fieldpath.MakePathOrDie("a")),
				Modified: fieldpath.NewSet(fieldpath.MakePathOrDie("b")),
				Removed:  fieldpath.NewSet(fieldpath.MakePathOrDie("c")),
			},
			Remove: fieldpath.NewSet(),
			Expect: &typed.Comparison{
				Added:    fieldpath.NewSet(fieldpath.MakePathOrDie("a")),
				Modified: fieldpath.NewSet(fieldpath.MakePathOrDie("b")),
				Removed:  fieldpath.NewSet(fieldpath.MakePathOrDie("c")),
			},
		},
		{
			name: "removes from entire object",
			Comparison: &typed.Comparison{
				Added:    fieldpath.NewSet(fieldpath.MakePathOrDie("a", "aa")),
				Modified: fieldpath.NewSet(fieldpath.MakePathOrDie("b", "ba")),
				Removed:  fieldpath.NewSet(fieldpath.MakePathOrDie("c", "ca")),
			},
			Remove: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a"),
				fieldpath.MakePathOrDie("b"),
				fieldpath.MakePathOrDie("c"),
			),
			Expect: &typed.Comparison{
				Added:    fieldpath.NewSet(),
				Modified: fieldpath.NewSet(),
				Removed:  fieldpath.NewSet(),
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			c.Comparison.ExcludeFields(c.Remove)
			if (!c.Comparison.Added.Equals(c.Expect.Added) ||
				!c.Comparison.Modified.Equals(c.Expect.Modified) ||
				!c.Comparison.Removed.Equals(c.Expect.Removed)) != c.Fails {
				t.Fatalf("remove expected: \n%v\nremoved:\n%v\ngot:\n%v\n", c.Expect, c.Remove, c.Comparison)
			}
		})
	}
}
