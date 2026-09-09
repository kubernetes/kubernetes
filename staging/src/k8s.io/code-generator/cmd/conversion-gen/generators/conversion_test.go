/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package generators

import (
	"testing"

	"k8s.io/gengo/v2/types"
)

func builtin(name string) *types.Type {
	return &types.Type{Name: types.Name{Name: name}, Kind: types.Builtin}
}

func structOf(pkg, name string, members ...types.Member) *types.Type {
	return &types.Type{
		Name:    types.Name{Package: pkg, Name: name},
		Kind:    types.Struct,
		Members: members,
	}
}

func field(name string, t *types.Type, comments ...string) types.Member {
	return types.Member{Name: name, Type: t, CommentLines: comments}
}

func conversionFn(comments ...string) *types.Type {
	return &types.Type{Name: types.Name{Name: "ConversionFunc"}, CommentLines: comments}
}

func TestCanUseMemoryCopyConversion(t *testing.T) {
	str := builtin("string")
	i := builtin("int")
	meta := structOf("meta", "TypeMeta", field("Kind", str))
	metaExt := structOf("ext", "TypeMeta", field("Kind", str))
	metaInt := structOf("int", "TypeMeta", field("Kind", str))
	metaAlias := &types.Type{Name: types.Name{Package: "ext", Name: "MetaAlias"}, Kind: types.Alias, Underlying: metaExt}

	level2Ext := structOf("ext", "Level2", field("Value", str))
	level2Int := structOf("int", "Level2", field("Value", str))
	level1Ext := structOf("ext", "Level1", field("Level2", level2Ext))
	level1Int := structOf("int", "Level1", field("Level2", level2Int))
	aSpecExt := structOf("ext", "ASpec", field("Level1", level1Ext))
	aSpecInt := structOf("int", "ASpec", field("Level1", level1Int))
	aExt := structOf("ext", "A", field("Spec", aSpecExt))
	aInt := structOf("int", "A", field("Spec", aSpecInt))

	tests := []struct {
		name              string
		outputPackage     string
		manualConversions conversionFuncMap
		in                *types.Type
		out               *types.Type
		want              bool
		wantErr           string
	}{
		{
			name: "identical",
			in:   structOf("ext", "T", field("Name", str), field("Age", i)),
			out:  structOf("int", "T", field("Name", str), field("Age", i)),
			want: true,
		},
		{
			name: "not identical",
			in:   structOf("ext", "T", field("Name", str)),
			out:  structOf("int", "T", field("Identifier", str)),
			want: false,
		},
		{
			name: "empty struct",
			in:   structOf("ext", "T"),
			out:  structOf("int", "T"),
			want: true,
		},
		{
			name: "member opted out of conversion generation",
			in:   structOf("ext", "T", field("Name", str, "+k8s:conversion-gen=false")),
			out:  structOf("int", "T", field("Name", str)),
			want: false,
		},
		{
			name: "member missing in peer",
			in:   structOf("ext", "T", field("Name", str), field("Extra", i)),
			out:  structOf("int", "T", field("Name", str)),
			want: false,
		},
		{
			name:          "unexported member, output package internal",
			outputPackage: "int",
			in:            structOf("ext", "T", field("secret", str)),
			out:           structOf("int", "T", field("secret", str)),
			want:          false,
		},
		{
			name:          "unexported member, output package is external",
			outputPackage: "ext",
			in:            structOf("ext", "T", field("secret", str)),
			out:           structOf("int", "T", field("secret", str)),
			want:          false,
		},
		{
			name:          "unexported member, but conversion happens inside package",
			outputPackage: "p",
			in:            structOf("p", "T", field("secret", str)),
			out:           structOf("p", "T", field("secret", str)),
			want:          true,
		},
		{
			name:              "member has a dropping manual conversion",
			manualConversions: conversionFuncMap{{meta, meta}: conversionFn("+k8s:conversion-fn=drop")},
			in:                structOf("ext", "T", field("Meta", meta)),
			out:               structOf("int", "T", field("Meta", meta)),
			want:              false,
		},
		{
			name:              "member has a manual conversion",
			manualConversions: conversionFuncMap{{meta, meta}: conversionFn()},
			in:                structOf("ext", "T", field("Meta", meta)),
			out:               structOf("int", "T", field("Meta", meta)),
			want:              false,
		},
		{
			name:              "member has a copy-only manual conversion",
			manualConversions: conversionFuncMap{{meta, meta}: conversionFn("+k8s:conversion-fn=copy-only")},
			in:                structOf("ext", "T", field("Meta", meta)),
			out:               structOf("int", "T", field("Meta", meta)),
			want:              true,
		},
		{
			name:              "identical types: drop is honored",
			manualConversions: conversionFuncMap{{metaExt, metaInt}: conversionFn("+k8s:conversion-fn=drop")},
			in:                structOf("ext", "T", field("Meta", metaExt)),
			out:               structOf("int", "T", field("Meta", metaInt)),
			want:              false,
		},
		{
			name:              "identical types: copy-only",
			manualConversions: conversionFuncMap{{metaExt, metaInt}: conversionFn("+k8s:conversion-fn=copy-only")},
			in:                structOf("ext", "T", field("Meta", metaExt)),
			out:               structOf("int", "T", field("Meta", metaInt)),
			want:              true,
		},
		{
			name:              "conversion on alias, not underlying type",
			manualConversions: conversionFuncMap{{metaAlias, metaAlias}: conversionFn("+k8s:conversion-fn=drop")},
			in:                structOf("ext", "T", field("Meta", metaAlias)),
			out:               structOf("int", "T", field("Meta", metaAlias)),
			want:              false,
		},
		{
			name:    "tag parse error",
			in:      structOf("ext", "T", field("Name", str, "+k8s:conversion-gen(a, b)=false")),
			out:     structOf("int", "T", field("Name", str)),
			wantErr: "multiple arguments must use 'name: value' syntax",
		},
		{
			name: "equalMemoryTypes compare: identical",
			in:   structOf("ext", "T", field("Name", str)),
			out:  structOf("int", "T", field("Name", str)),
			want: true,
		},
		{
			name: "equalMemoryTypes compare: not identical",
			in:   structOf("ext", "T", field("Name", str)),
			out:  structOf("int", "T", field("Name", i)),
			want: false,
		},
		{
			name: "two members: not identical",
			in:   structOf("ext", "T", field("OK", str), field("Bad", i, "+k8s:conversion-gen=false")),
			out:  structOf("int", "T", field("OK", str), field("Bad", i)),
			want: false,
		},
		{
			name: "conversion-gen=true does not opt out",
			in:   structOf("ext", "T", field("Name", str, "+k8s:conversion-gen=true")),
			out:  structOf("int", "T", field("Name", str)),
			want: true,
		},
		{
			name:              "nested manual conversion blocks memory copy at every ancestor",
			manualConversions: conversionFuncMap{{level2Ext, level2Int}: conversionFn()},
			in:                aExt,
			out:               aInt,
			want:              false,
		},
		{
			name:              "nested copy-only conversion does not block memory copy",
			manualConversions: conversionFuncMap{{level2Ext, level2Int}: conversionFn("+k8s:conversion-fn=copy-only")},
			in:                aExt,
			out:               aInt,
			want:              true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			useUnsafe := equalMemoryTypes{}
			for pair, fn := range tt.manualConversions {
				copyOnly, _ := isCopyOnly(fn.CommentLines)
				if copyOnly {
					continue
				}
				useUnsafe.Skip(pair.inType, pair.outType)
			}
			g := &genConversion{
				outputPackage:     tt.outputPackage,
				manualConversions: tt.manualConversions,
				useUnsafe:         useUnsafe,
			}
			gotOK, err := g.canUseMemoryCopyConversion(tt.in, tt.out)
			if err != nil && err.Error() != tt.wantErr {
				t.Fatalf("canUseMemoryCopyConversion() error = %v, wantErr = %v", err, tt.wantErr)
			}
			if gotOK != tt.want {
				t.Errorf("canUseMemoryCopyConversion() = %v, want %v", gotOK, tt.want)
			}
		})
	}
}
