// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"io/ioutil"
	"strings"
	"testing"
)

var testZipData = func() []byte {
	if data, err := ioutil.ReadFile("../examples/local.zip"); err == nil {
		return data
	}
	return nil
}()

func TestGettext(t *testing.T) {
	Textdomain("hello")

	// local file system
	BindTextdomain("hello", "../examples/local", nil)
	testGettext(t, true)
	BindTextdomain("hello", "", nil)
	testGettext(t, false)

	// local zip file system
	BindTextdomain("hello", "../examples/local.zip", nil)
	testGettext(t, true)
	BindTextdomain("hello", "", nil)
	testGettext(t, false)

	// embedded zip file system
	BindTextdomain("hello", "local.zip", testZipData)
	testGettext(t, true)
	BindTextdomain("hello", "", nil)
	testGettext(t, false)
}

func TestGetdata(t *testing.T) {
	Textdomain("hello")

	// local file system
	BindTextdomain("hello", "../examples/local", nil)
	testGetdata(t, true)
	BindTextdomain("hello", "", nil)
	testGetdata(t, false)

	// local zip file system
	BindTextdomain("hello", "../examples/local.zip", nil)
	testGetdata(t, true)
	BindTextdomain("hello", "", nil)
	testGetdata(t, false)

	// embedded zip file system
	BindTextdomain("hello", "local.zip", testZipData)
	testGetdata(t, true)
	BindTextdomain("hello", "", nil)
	testGetdata(t, false)
}

func testGettext(t *testing.T, hasTransle bool) {
	for i, v := range testTexts {
		if lang := SetLocale(v.lang); lang != v.lang {
			t.Fatalf("%d: expect = %s, got = %v", i, v.lang, lang)
		}
		if hasTransle {
			if dst := PGettext(v.ctx, v.src); dst != v.dst {
				t.Fatalf("%d: expect = %q, got = %q", i, v.dst, dst)
			}
		} else {
			if dst := PGettext(v.ctx, v.src); dst != v.src {
				t.Fatalf("%d: expect = %s, got = %v", i, v.src, dst)
			}
		}
	}
}

func testGetdata(t *testing.T, hasTransle bool) {
	for i, v := range testResources {
		if lang := SetLocale(v.lang); lang != v.lang {
			t.Fatalf("%d: expect = %s, got = %v", i, v.lang, lang)
		}
		if hasTransle {
			v.data = strings.Replace(v.data, "\r", "", -1)
			data := strings.Replace(string(Getdata(v.path)), "\r", "", -1)
			if data != v.data {
				t.Fatalf("%d: expect = %q, got = %q", i, v.data, data)
			}
		} else {
			if data := string(Getdata(v.path)); data != "" {
				t.Fatalf("%d: expect = %s, got = %v", i, "", data)
			}
		}
	}
}

func BenchmarkGettext(b *testing.B) {
	SetLocale("zh_CN")
	BindTextdomain("hello", "../examples/local", nil)
	Textdomain("hello")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PGettext(testTexts[0].ctx, testTexts[0].src)
	}
}
func BenchmarkGettext_Zip(b *testing.B) {
	SetLocale("zh_CN")
	BindTextdomain("hello", "../examples/local.zip", nil)
	Textdomain("hello")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PGettext(testTexts[0].ctx, testTexts[0].src)
	}
}

func BenchmarkGetdata(b *testing.B) {
	SetLocale("zh_CN")
	BindTextdomain("hello", "../examples/local", nil)
	Textdomain("hello")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Getdata(testResources[0].path)
	}
}
func BenchmarkGetdata_Zip(b *testing.B) {
	SetLocale("zh_CN")
	BindTextdomain("hello", "../examples/local.zip", nil)
	Textdomain("hello")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Getdata(testResources[0].path)
	}
}

var testTexts = []struct {
	lang string
	ctx  string
	src  string
	dst  string
}{
	// default
	{"default", "main.init", "Gettext in init.", "Gettext in init."},
	{"default", "main.main", "Hello, world!", "Hello, world!"},
	{"default", "main.func", "Gettext in func.", "Gettext in func."},
	{"default", "github.com/chai2010/gettext-go/examples/hi.SayHi", "pkg hi: Hello, world!", "pkg hi: Hello, world!"},

	// zh_CN
	{"zh_CN", "main.init", "Gettext in init.", "Init函数中的Gettext."},
	{"zh_CN", "main.main", "Hello, world!", "你好, 世界!"},
	{"zh_CN", "main.func", "Gettext in func.", "闭包函数中的Gettext."},
	{"zh_CN", "github.com/chai2010/gettext-go/examples/hi.SayHi", "pkg hi: Hello, world!", "来自\"Hi\"包的问候: 你好, 世界!"},

	// zh_TW
	{"zh_TW", "main.init", "Gettext in init.", "Init函數中的Gettext."},
	{"zh_TW", "main.main", "Hello, world!", "你好, 世界!"},
	{"zh_TW", "main.func", "Gettext in func.", "閉包函數中的Gettext."},
	{"zh_TW", "github.com/chai2010/gettext-go/examples/hi.SayHi", "pkg hi: Hello, world!", "來自\"Hi\"包的問候: 你好, 世界!"},
}

var testResources = []struct {
	lang string
	path string
	data string
}{
	// default
	{
		"default",
		"poems.txt",
		`Drinking Alone Under the Moon
Li Bai

flowers among one jar liquor
alone carouse without mutual intimate

raise cup greet bright moon
facing shadow become three persons

moon since not free to-drink
shadow follow accompany my body

briefly accompany moon with shadow
go happy should avail-oneself-of spring

my song moon walk-to-and-fro irresolute
my dance shadow fragments disorderly

sober time together mix glad
drunk after each divide scatter

eternal connect without consciouness-of-self roam
mutual appointment remote cloud Milky-Way
`,
	},

	// zh_CN
	{
		"zh_CN",
		"poems.txt",
		`yuèxiàdúzhuó
月下独酌
lǐbái
李白

huājiānyīhújiǔ，dúzhuówúxiānɡqīn。
花间一壶酒，独酌无相亲。
jǔbēiyāomínɡyuè，duìyǐnɡchénɡsānrén。
举杯邀明月，对影成三人。
yuèjìbùjiěyǐn，yǐnɡtúsuíwǒshēn。
月既不解饮，影徒随我身。
zànbànyuèjiānɡyǐnɡ，xínɡlèxūjíchūn。
暂伴月将影，行乐须及春。
wǒɡēyuèpáihuái，wǒwǔyǐnɡlínɡluàn。
我歌月徘徊，我舞影零乱。
xǐnɡshítónɡjiāohuān，zuìhòuɡèfēnsàn。
醒时同交欢，醉后各分散。
yǒnɡjiéwúqínɡyóu，xiānɡqīmiǎoyúnhàn。
永结无情游，相期邈云汉。
`,
	},

	// zh_TW
	{
		"zh_TW",
		"poems.txt",
		`yuèxiàdúzhuó
月下獨酌
lǐbái
李白

huājiānyīhújiǔ，dúzhuówúxiānɡqīn。
花間一壺酒，獨酌無相親。
jǔbēiyāomínɡyuè，duìyǐnɡchénɡsānrén。
舉杯邀明月，對影成三人。
yuèjìbùjiěyǐn，yǐnɡtúsuíwǒshēn。
月既不解飲，影徒隨我身。
zànbànyuèjiānɡyǐnɡ，xínɡlèxūjíchūn。
暫伴月將影，行樂須及春。
wǒɡēyuèpáihuái，wǒwǔyǐnɡlínɡluàn。
我歌月徘徊，我舞影零亂。
xǐnɡshítónɡjiāohuān，zuìhòuɡèfēnsàn。
醒時同交歡，醉後各分散。
yǒnɡjiéwúqínɡyóu，xiānɡqīmiǎoyúnhàn。
永結無情遊，相期邈雲漢。
`,
	},
}
