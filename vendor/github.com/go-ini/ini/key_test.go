// Copyright 2014 Unknwon
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package ini

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func Test_Key(t *testing.T) {
	Convey("Test getting and setting values", t, func() {
		cfg, err := Load([]byte(_CONF_DATA), "testdata/conf.ini")
		So(err, ShouldBeNil)
		So(cfg, ShouldNotBeNil)

		Convey("Get values in default section", func() {
			sec := cfg.Section("")
			So(sec, ShouldNotBeNil)
			So(sec.Key("NAME").Value(), ShouldEqual, "ini")
			So(sec.Key("NAME").String(), ShouldEqual, "ini")
			So(sec.Key("NAME").Validate(func(in string) string {
				return in
			}), ShouldEqual, "ini")
			So(sec.Key("NAME").Comment, ShouldEqual, "; Package name")
			So(sec.Key("IMPORT_PATH").String(), ShouldEqual, "gopkg.in/ini.v1")
		})

		Convey("Get values in non-default section", func() {
			sec := cfg.Section("author")
			So(sec, ShouldNotBeNil)
			So(sec.Key("NAME").String(), ShouldEqual, "Unknwon")
			So(sec.Key("GITHUB").String(), ShouldEqual, "https://github.com/Unknwon")

			sec = cfg.Section("package")
			So(sec, ShouldNotBeNil)
			So(sec.Key("CLONE_URL").String(), ShouldEqual, "https://gopkg.in/ini.v1")
		})

		Convey("Get auto-increment key names", func() {
			keys := cfg.Section("features").Keys()
			for i, k := range keys {
				So(k.Name(), ShouldEqual, fmt.Sprintf("#%d", i+1))
			}
		})

		Convey("Get parent-keys that are available to the child section", func() {
			parentKeys := cfg.Section("package.sub").ParentKeys()
			for _, k := range parentKeys {
				So(k.Name(), ShouldEqual, "CLONE_URL")
			}
		})

		Convey("Get overwrite value", func() {
			So(cfg.Section("author").Key("E-MAIL").String(), ShouldEqual, "u@gogs.io")
		})

		Convey("Get sections", func() {
			sections := cfg.Sections()
			for i, name := range []string{DEFAULT_SECTION, "author", "package", "package.sub", "features", "types", "array", "note", "comments", "advance"} {
				So(sections[i].Name(), ShouldEqual, name)
			}
		})

		Convey("Get parent section value", func() {
			So(cfg.Section("package.sub").Key("CLONE_URL").String(), ShouldEqual, "https://gopkg.in/ini.v1")
			So(cfg.Section("package.fake.sub").Key("CLONE_URL").String(), ShouldEqual, "https://gopkg.in/ini.v1")
		})

		Convey("Get multiple line value", func() {
			So(cfg.Section("author").Key("BIO").String(), ShouldEqual, "Gopher.\nCoding addict.\nGood man.\n")
		})

		Convey("Get values with type", func() {
			sec := cfg.Section("types")
			v1, err := sec.Key("BOOL").Bool()
			So(err, ShouldBeNil)
			So(v1, ShouldBeTrue)

			v1, err = sec.Key("BOOL_FALSE").Bool()
			So(err, ShouldBeNil)
			So(v1, ShouldBeFalse)

			v2, err := sec.Key("FLOAT64").Float64()
			So(err, ShouldBeNil)
			So(v2, ShouldEqual, 1.25)

			v3, err := sec.Key("INT").Int()
			So(err, ShouldBeNil)
			So(v3, ShouldEqual, 10)

			v4, err := sec.Key("INT").Int64()
			So(err, ShouldBeNil)
			So(v4, ShouldEqual, 10)

			v5, err := sec.Key("UINT").Uint()
			So(err, ShouldBeNil)
			So(v5, ShouldEqual, 3)

			v6, err := sec.Key("UINT").Uint64()
			So(err, ShouldBeNil)
			So(v6, ShouldEqual, 3)

			t, err := time.Parse(time.RFC3339, "2015-01-01T20:17:05Z")
			So(err, ShouldBeNil)
			v7, err := sec.Key("TIME").Time()
			So(err, ShouldBeNil)
			So(v7.String(), ShouldEqual, t.String())

			Convey("Must get values with type", func() {
				So(sec.Key("STRING").MustString("404"), ShouldEqual, "str")
				So(sec.Key("BOOL").MustBool(), ShouldBeTrue)
				So(sec.Key("FLOAT64").MustFloat64(), ShouldEqual, 1.25)
				So(sec.Key("INT").MustInt(), ShouldEqual, 10)
				So(sec.Key("INT").MustInt64(), ShouldEqual, 10)
				So(sec.Key("UINT").MustUint(), ShouldEqual, 3)
				So(sec.Key("UINT").MustUint64(), ShouldEqual, 3)
				So(sec.Key("TIME").MustTime().String(), ShouldEqual, t.String())

				dur, err := time.ParseDuration("2h45m")
				So(err, ShouldBeNil)
				So(sec.Key("DURATION").MustDuration().Seconds(), ShouldEqual, dur.Seconds())

				Convey("Must get values with default value", func() {
					So(sec.Key("STRING_404").MustString("404"), ShouldEqual, "404")
					So(sec.Key("BOOL_404").MustBool(true), ShouldBeTrue)
					So(sec.Key("FLOAT64_404").MustFloat64(2.5), ShouldEqual, 2.5)
					So(sec.Key("INT_404").MustInt(15), ShouldEqual, 15)
					So(sec.Key("INT64_404").MustInt64(15), ShouldEqual, 15)
					So(sec.Key("UINT_404").MustUint(6), ShouldEqual, 6)
					So(sec.Key("UINT64_404").MustUint64(6), ShouldEqual, 6)

					t, err := time.Parse(time.RFC3339, "2014-01-01T20:17:05Z")
					So(err, ShouldBeNil)
					So(sec.Key("TIME_404").MustTime(t).String(), ShouldEqual, t.String())

					So(sec.Key("DURATION_404").MustDuration(dur).Seconds(), ShouldEqual, dur.Seconds())

					Convey("Must should set default as key value", func() {
						So(sec.Key("STRING_404").String(), ShouldEqual, "404")
						So(sec.Key("BOOL_404").String(), ShouldEqual, "true")
						So(sec.Key("FLOAT64_404").String(), ShouldEqual, "2.5")
						So(sec.Key("INT_404").String(), ShouldEqual, "15")
						So(sec.Key("INT64_404").String(), ShouldEqual, "15")
						So(sec.Key("UINT_404").String(), ShouldEqual, "6")
						So(sec.Key("UINT64_404").String(), ShouldEqual, "6")
						So(sec.Key("TIME_404").String(), ShouldEqual, "2014-01-01T20:17:05Z")
						So(sec.Key("DURATION_404").String(), ShouldEqual, "2h45m0s")
					})
				})
			})
		})

		Convey("Get value with candidates", func() {
			sec := cfg.Section("types")
			So(sec.Key("STRING").In("", []string{"str", "arr", "types"}), ShouldEqual, "str")
			So(sec.Key("FLOAT64").InFloat64(0, []float64{1.25, 2.5, 3.75}), ShouldEqual, 1.25)
			So(sec.Key("INT").InInt(0, []int{10, 20, 30}), ShouldEqual, 10)
			So(sec.Key("INT").InInt64(0, []int64{10, 20, 30}), ShouldEqual, 10)
			So(sec.Key("UINT").InUint(0, []uint{3, 6, 9}), ShouldEqual, 3)
			So(sec.Key("UINT").InUint64(0, []uint64{3, 6, 9}), ShouldEqual, 3)

			zt, err := time.Parse(time.RFC3339, "0001-01-01T01:00:00Z")
			So(err, ShouldBeNil)
			t, err := time.Parse(time.RFC3339, "2015-01-01T20:17:05Z")
			So(err, ShouldBeNil)
			So(sec.Key("TIME").InTime(zt, []time.Time{t, time.Now(), time.Now().Add(1 * time.Second)}).String(), ShouldEqual, t.String())

			Convey("Get value with candidates and default value", func() {
				So(sec.Key("STRING_404").In("str", []string{"str", "arr", "types"}), ShouldEqual, "str")
				So(sec.Key("FLOAT64_404").InFloat64(1.25, []float64{1.25, 2.5, 3.75}), ShouldEqual, 1.25)
				So(sec.Key("INT_404").InInt(10, []int{10, 20, 30}), ShouldEqual, 10)
				So(sec.Key("INT64_404").InInt64(10, []int64{10, 20, 30}), ShouldEqual, 10)
				So(sec.Key("UINT_404").InUint(3, []uint{3, 6, 9}), ShouldEqual, 3)
				So(sec.Key("UINT_404").InUint64(3, []uint64{3, 6, 9}), ShouldEqual, 3)
				So(sec.Key("TIME_404").InTime(t, []time.Time{time.Now(), time.Now(), time.Now().Add(1 * time.Second)}).String(), ShouldEqual, t.String())
			})
		})

		Convey("Get values in range", func() {
			sec := cfg.Section("types")
			So(sec.Key("FLOAT64").RangeFloat64(0, 1, 2), ShouldEqual, 1.25)
			So(sec.Key("INT").RangeInt(0, 10, 20), ShouldEqual, 10)
			So(sec.Key("INT").RangeInt64(0, 10, 20), ShouldEqual, 10)

			minT, err := time.Parse(time.RFC3339, "0001-01-01T01:00:00Z")
			So(err, ShouldBeNil)
			midT, err := time.Parse(time.RFC3339, "2013-01-01T01:00:00Z")
			So(err, ShouldBeNil)
			maxT, err := time.Parse(time.RFC3339, "9999-01-01T01:00:00Z")
			So(err, ShouldBeNil)
			t, err := time.Parse(time.RFC3339, "2015-01-01T20:17:05Z")
			So(err, ShouldBeNil)
			So(sec.Key("TIME").RangeTime(t, minT, maxT).String(), ShouldEqual, t.String())

			Convey("Get value in range with default value", func() {
				So(sec.Key("FLOAT64").RangeFloat64(5, 0, 1), ShouldEqual, 5)
				So(sec.Key("INT").RangeInt(7, 0, 5), ShouldEqual, 7)
				So(sec.Key("INT").RangeInt64(7, 0, 5), ShouldEqual, 7)
				So(sec.Key("TIME").RangeTime(t, minT, midT).String(), ShouldEqual, t.String())
			})
		})

		Convey("Get values into slice", func() {
			sec := cfg.Section("array")
			So(strings.Join(sec.Key("STRINGS").Strings(","), ","), ShouldEqual, "en,zh,de")
			So(len(sec.Key("STRINGS_404").Strings(",")), ShouldEqual, 0)

			vals1 := sec.Key("FLOAT64S").Float64s(",")
			float64sEqual(vals1, 1.1, 2.2, 3.3)

			vals2 := sec.Key("INTS").Ints(",")
			intsEqual(vals2, 1, 2, 3)

			vals3 := sec.Key("INTS").Int64s(",")
			int64sEqual(vals3, 1, 2, 3)

			vals4 := sec.Key("UINTS").Uints(",")
			uintsEqual(vals4, 1, 2, 3)

			vals5 := sec.Key("UINTS").Uint64s(",")
			uint64sEqual(vals5, 1, 2, 3)

			t, err := time.Parse(time.RFC3339, "2015-01-01T20:17:05Z")
			So(err, ShouldBeNil)
			vals6 := sec.Key("TIMES").Times(",")
			timesEqual(vals6, t, t, t)
		})

		Convey("Get valid values into slice", func() {
			sec := cfg.Section("array")
			vals1 := sec.Key("FLOAT64S").ValidFloat64s(",")
			float64sEqual(vals1, 1.1, 2.2, 3.3)

			vals2 := sec.Key("INTS").ValidInts(",")
			intsEqual(vals2, 1, 2, 3)

			vals3 := sec.Key("INTS").ValidInt64s(",")
			int64sEqual(vals3, 1, 2, 3)

			vals4 := sec.Key("UINTS").ValidUints(",")
			uintsEqual(vals4, 1, 2, 3)

			vals5 := sec.Key("UINTS").ValidUint64s(",")
			uint64sEqual(vals5, 1, 2, 3)

			t, err := time.Parse(time.RFC3339, "2015-01-01T20:17:05Z")
			So(err, ShouldBeNil)
			vals6 := sec.Key("TIMES").ValidTimes(",")
			timesEqual(vals6, t, t, t)
		})

		Convey("Get values one type into slice of another type", func() {
			sec := cfg.Section("array")
			vals1 := sec.Key("STRINGS").ValidFloat64s(",")
			So(vals1, ShouldBeEmpty)

			vals2 := sec.Key("STRINGS").ValidInts(",")
			So(vals2, ShouldBeEmpty)

			vals3 := sec.Key("STRINGS").ValidInt64s(",")
			So(vals3, ShouldBeEmpty)

			vals4 := sec.Key("STRINGS").ValidUints(",")
			So(vals4, ShouldBeEmpty)

			vals5 := sec.Key("STRINGS").ValidUint64s(",")
			So(vals5, ShouldBeEmpty)

			vals6 := sec.Key("STRINGS").ValidTimes(",")
			So(vals6, ShouldBeEmpty)
		})

		Convey("Get valid values into slice without errors", func() {
			sec := cfg.Section("array")
			vals1, err := sec.Key("FLOAT64S").StrictFloat64s(",")
			So(err, ShouldBeNil)
			float64sEqual(vals1, 1.1, 2.2, 3.3)

			vals2, err := sec.Key("INTS").StrictInts(",")
			So(err, ShouldBeNil)
			intsEqual(vals2, 1, 2, 3)

			vals3, err := sec.Key("INTS").StrictInt64s(",")
			So(err, ShouldBeNil)
			int64sEqual(vals3, 1, 2, 3)

			vals4, err := sec.Key("UINTS").StrictUints(",")
			So(err, ShouldBeNil)
			uintsEqual(vals4, 1, 2, 3)

			vals5, err := sec.Key("UINTS").StrictUint64s(",")
			So(err, ShouldBeNil)
			uint64sEqual(vals5, 1, 2, 3)

			t, err := time.Parse(time.RFC3339, "2015-01-01T20:17:05Z")
			So(err, ShouldBeNil)
			vals6, err := sec.Key("TIMES").StrictTimes(",")
			So(err, ShouldBeNil)
			timesEqual(vals6, t, t, t)
		})

		Convey("Get invalid values into slice", func() {
			sec := cfg.Section("array")
			vals1, err := sec.Key("STRINGS").StrictFloat64s(",")
			So(vals1, ShouldBeEmpty)
			So(err, ShouldNotBeNil)

			vals2, err := sec.Key("STRINGS").StrictInts(",")
			So(vals2, ShouldBeEmpty)
			So(err, ShouldNotBeNil)

			vals3, err := sec.Key("STRINGS").StrictInt64s(",")
			So(vals3, ShouldBeEmpty)
			So(err, ShouldNotBeNil)

			vals4, err := sec.Key("STRINGS").StrictUints(",")
			So(vals4, ShouldBeEmpty)
			So(err, ShouldNotBeNil)

			vals5, err := sec.Key("STRINGS").StrictUint64s(",")
			So(vals5, ShouldBeEmpty)
			So(err, ShouldNotBeNil)

			vals6, err := sec.Key("STRINGS").StrictTimes(",")
			So(vals6, ShouldBeEmpty)
			So(err, ShouldNotBeNil)
		})

		Convey("Get key hash", func() {
			cfg.Section("").KeysHash()
		})

		Convey("Set key value", func() {
			k := cfg.Section("author").Key("NAME")
			k.SetValue("无闻")
			So(k.String(), ShouldEqual, "无闻")
		})

		Convey("Get key strings", func() {
			So(strings.Join(cfg.Section("types").KeyStrings(), ","), ShouldEqual, "STRING,BOOL,BOOL_FALSE,FLOAT64,INT,TIME,DURATION,UINT")
		})

		Convey("Delete a key", func() {
			cfg.Section("package.sub").DeleteKey("UNUSED_KEY")
			_, err := cfg.Section("package.sub").GetKey("UNUSED_KEY")
			So(err, ShouldNotBeNil)
		})

		Convey("Has Key (backwards compatible)", func() {
			sec := cfg.Section("package.sub")
			haskey1 := sec.Haskey("UNUSED_KEY")
			haskey2 := sec.Haskey("CLONE_URL")
			haskey3 := sec.Haskey("CLONE_URL_NO")
			So(haskey1, ShouldBeTrue)
			So(haskey2, ShouldBeTrue)
			So(haskey3, ShouldBeFalse)
		})

		Convey("Has Key", func() {
			sec := cfg.Section("package.sub")
			haskey1 := sec.HasKey("UNUSED_KEY")
			haskey2 := sec.HasKey("CLONE_URL")
			haskey3 := sec.HasKey("CLONE_URL_NO")
			So(haskey1, ShouldBeTrue)
			So(haskey2, ShouldBeTrue)
			So(haskey3, ShouldBeFalse)
		})

		Convey("Has Value", func() {
			sec := cfg.Section("author")
			hasvalue1 := sec.HasValue("Unknwon")
			hasvalue2 := sec.HasValue("doc")
			So(hasvalue1, ShouldBeTrue)
			So(hasvalue2, ShouldBeFalse)
		})
	})

	Convey("Test getting and setting bad values", t, func() {
		cfg, err := Load([]byte(_CONF_DATA), "testdata/conf.ini")
		So(err, ShouldBeNil)
		So(cfg, ShouldNotBeNil)

		Convey("Create new key with empty name", func() {
			k, err := cfg.Section("").NewKey("", "")
			So(err, ShouldNotBeNil)
			So(k, ShouldBeNil)
		})

		Convey("Create new section with empty name", func() {
			s, err := cfg.NewSection("")
			So(err, ShouldNotBeNil)
			So(s, ShouldBeNil)
		})

		Convey("Create new sections with empty name", func() {
			So(cfg.NewSections(""), ShouldNotBeNil)
		})

		Convey("Get section that not exists", func() {
			s, err := cfg.GetSection("404")
			So(err, ShouldNotBeNil)
			So(s, ShouldBeNil)

			s = cfg.Section("404")
			So(s, ShouldNotBeNil)
		})
	})

	Convey("Test key hash clone", t, func() {
		cfg, err := Load([]byte(strings.Replace("network=tcp,addr=127.0.0.1:6379,db=4,pool_size=100,idle_timeout=180", ",", "\n", -1)))
		So(err, ShouldBeNil)
		for _, v := range cfg.Section("").KeysHash() {
			So(len(v), ShouldBeGreaterThan, 0)
		}
	})

	Convey("Key has empty value", t, func() {
		_conf := `key1=
key2= ; comment`
		cfg, err := Load([]byte(_conf))
		So(err, ShouldBeNil)
		So(cfg.Section("").Key("key1").Value(), ShouldBeEmpty)
	})
}

const _CONF_GIT_CONFIG = `
[remote "origin"]
        url = https://github.com/Antergone/test1.git
        url = https://github.com/Antergone/test2.git
`

func Test_Key_Shadows(t *testing.T) {
	Convey("Shadows keys", t, func() {
		Convey("Disable shadows", func() {
			cfg, err := Load([]byte(_CONF_GIT_CONFIG))
			So(err, ShouldBeNil)
			So(cfg.Section(`remote "origin"`).Key("url").String(), ShouldEqual, "https://github.com/Antergone/test2.git")
		})

		Convey("Enable shadows", func() {
			cfg, err := ShadowLoad([]byte(_CONF_GIT_CONFIG))
			So(err, ShouldBeNil)
			So(cfg.Section(`remote "origin"`).Key("url").String(), ShouldEqual, "https://github.com/Antergone/test1.git")
			So(strings.Join(cfg.Section(`remote "origin"`).Key("url").ValueWithShadows(), " "), ShouldEqual,
				"https://github.com/Antergone/test1.git https://github.com/Antergone/test2.git")

			Convey("Save with shadows", func() {
				var buf bytes.Buffer
				_, err := cfg.WriteTo(&buf)
				So(err, ShouldBeNil)
				So(buf.String(), ShouldEqual, `[remote "origin"]
url = https://github.com/Antergone/test1.git
url = https://github.com/Antergone/test2.git

`)
			})
		})
	})
}

func newTestFile(block bool) *File {
	c, _ := Load([]byte(_CONF_DATA))
	c.BlockMode = block
	return c
}

func Benchmark_Key_Value(b *testing.B) {
	c := newTestFile(true)
	for i := 0; i < b.N; i++ {
		c.Section("").Key("NAME").Value()
	}
}

func Benchmark_Key_Value_NonBlock(b *testing.B) {
	c := newTestFile(false)
	for i := 0; i < b.N; i++ {
		c.Section("").Key("NAME").Value()
	}
}

func Benchmark_Key_Value_ViaSection(b *testing.B) {
	c := newTestFile(true)
	sec := c.Section("")
	for i := 0; i < b.N; i++ {
		sec.Key("NAME").Value()
	}
}

func Benchmark_Key_Value_ViaSection_NonBlock(b *testing.B) {
	c := newTestFile(false)
	sec := c.Section("")
	for i := 0; i < b.N; i++ {
		sec.Key("NAME").Value()
	}
}

func Benchmark_Key_Value_Direct(b *testing.B) {
	c := newTestFile(true)
	key := c.Section("").Key("NAME")
	for i := 0; i < b.N; i++ {
		key.Value()
	}
}

func Benchmark_Key_Value_Direct_NonBlock(b *testing.B) {
	c := newTestFile(false)
	key := c.Section("").Key("NAME")
	for i := 0; i < b.N; i++ {
		key.Value()
	}
}

func Benchmark_Key_String(b *testing.B) {
	c := newTestFile(true)
	for i := 0; i < b.N; i++ {
		_ = c.Section("").Key("NAME").String()
	}
}

func Benchmark_Key_String_NonBlock(b *testing.B) {
	c := newTestFile(false)
	for i := 0; i < b.N; i++ {
		_ = c.Section("").Key("NAME").String()
	}
}

func Benchmark_Key_String_ViaSection(b *testing.B) {
	c := newTestFile(true)
	sec := c.Section("")
	for i := 0; i < b.N; i++ {
		_ = sec.Key("NAME").String()
	}
}

func Benchmark_Key_String_ViaSection_NonBlock(b *testing.B) {
	c := newTestFile(false)
	sec := c.Section("")
	for i := 0; i < b.N; i++ {
		_ = sec.Key("NAME").String()
	}
}

func Benchmark_Key_SetValue(b *testing.B) {
	c := newTestFile(true)
	for i := 0; i < b.N; i++ {
		c.Section("").Key("NAME").SetValue("10")
	}
}

func Benchmark_Key_SetValue_VisSection(b *testing.B) {
	c := newTestFile(true)
	sec := c.Section("")
	for i := 0; i < b.N; i++ {
		sec.Key("NAME").SetValue("10")
	}
}
