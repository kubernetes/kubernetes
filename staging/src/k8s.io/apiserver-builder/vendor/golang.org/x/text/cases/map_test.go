// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cases

import (
	"bytes"
	"strings"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

type testCase struct {
	lang  string
	src   interface{} // string, []string, or nil to skip test
	title interface{} // string, []string, or nil to skip test
	lower interface{} // string, []string, or nil to skip test
	upper interface{} // string, []string, or nil to skip test
	opts  options
}

// We don't support the NoFinalSigma option, but we use it to test the
// underlying lower casers and to be able to compare differences in performance.
func noFinalSigma(o *options) {
	o.noFinalSigma = true
}

var testCases = []testCase{
	0: {
		lang:  "und",
		src:   "abc aBc ABC abC İsıI ΕΣΆΣ",
		title: "Abc Abc Abc Abc İsıi Εσάσ",
		lower: "abc abc abc abc i\u0307sıi εσάσ",
		upper: "ABC ABC ABC ABC İSII ΕΣΆΣ",
		opts:  getOpts(noFinalSigma),
	},

	1: {
		lang:  "und",
		src:   "abc aBc ABC abC İsıI ΕΣΆΣ Σ _Σ -Σ",
		title: "Abc Abc Abc Abc İsıi Εσάς Σ _Σ -Σ",
		lower: "abc abc abc abc i\u0307sıi εσάς σ _σ -σ",
		upper: "ABC ABC ABC ABC İSII ΕΣΆΣ Σ _Σ -Σ",
	},

	2: { // Title cased runes.
		lang:  supported,
		src:   "ǅA",
		title: "ǅa",
		lower: "ǆa",
		upper: "ǄA",
	},

	3: {
		// Title breaking.
		lang: supported,
		src: []string{
			"FOO CASE TEST",
			"DON'T DO THiS",
			"χωΡΊΣ χωΡΊΣ^a χωΡΊΣ:a χωΡΊΣ:^a χωΡΊΣ^ όμΩΣ Σ",
			"with-hyphens",
			"49ers 49ers",
			`"capitalize a^a -hyphen 0X _u a_u:a`,
			"MidNumLet a.b\u2018c\u2019d\u2024e\ufe52f\uff07f\uff0eg",
			"MidNum a,b;c\u037ed\u0589e\u060cf\u2044g\ufe50h",
			"\u0345 x\u3031x x\u05d0x \u05d0x a'.a a.a a4,a",
		},
		title: []string{
			"Foo Case Test",
			"Don't Do This",
			"Χωρίς Χωρίσ^A Χωρίσ:a Χωρίσ:^A Χωρίς^ Όμως Σ",
			"With-Hyphens",
			// Note that 49Ers is correct according to the spec.
			// TODO: provide some option to the user to treat different
			// characters as cased.
			"49Ers 49Ers",
			`"Capitalize A^A -Hyphen 0X _U A_u:a`,
			"Midnumlet A.b\u2018c\u2019d\u2024e\ufe52f\uff07f\uff0eg",
			"Midnum A,B;C\u037eD\u0589E\u060cF\u2044G\ufe50H",
			"\u0399 X\u3031X X\u05d0x \u05d0X A'.A A.a A4,A",
		},
	},

	// TODO: These are known deviations from the options{} Unicode Word Breaking
	// Algorithm.
	// {
	// 	"und",
	// 	"x_\u3031_x a4,4a",
	// 	"X_\u3031_x A4,4a", // Currently is "X_\U3031_X A4,4A".
	// 	"x_\u3031_x a4,4a",
	// 	"X_\u3031_X A4,4A",
	// 	options{},
	// },

	4: {
		// Tests title options
		lang:  "und",
		src:   "abc aBc ABC abC İsıI o'Brien",
		title: "Abc ABc ABC AbC İsıI O'Brien",
		opts:  getOpts(NoLower),
	},

	5: {
		lang:  "el",
		src:   "aBc ΟΔΌΣ Οδός Σο ΣΟ Σ oΣ ΟΣ σ ἕξ \u03ac",
		title: "Abc Οδός Οδός Σο Σο Σ Oς Ος Σ Ἕξ \u0386",
		lower: "abc οδός οδός σο σο σ oς ος σ ἕξ \u03ac",
		upper: "ABC ΟΔΟΣ ΟΔΟΣ ΣΟ ΣΟ Σ OΣ ΟΣ Σ ΕΞ \u0391", // Uppercase removes accents
	},

	6: {
		lang:  "tr az",
		src:   "Isiİ İsıI I\u0307sIiİ İsıI\u0307 I\u0300\u0307",
		title: "Isii İsıı I\u0307sıii İsıi I\u0300\u0307",
		lower: "ısii isıı isıii isıi \u0131\u0300\u0307",
		upper: "ISİİ İSII I\u0307SIİİ İSII\u0307 I\u0300\u0307",
	},

	7: {
		lang:  "lt",
		src:   "I Ï J J̈ Į Į̈ Ì Í Ĩ xi̇̈ xj̇̈ xį̇̈ xi̇̀ xi̇́ xi̇̃ XI XÏ XJ XJ̈ XĮ XĮ̈ XI̟̤",
		title: "I Ï J J̈ Į Į̈ Ì Í Ĩ Xi̇̈ Xj̇̈ Xį̇̈ Xi̇̀ Xi̇́ Xi̇̃ Xi Xi̇̈ Xj Xj̇̈ Xį Xį̇̈ Xi̟̤",
		lower: "i i̇̈ j j̇̈ į į̇̈ i̇̀ i̇́ i̇̃ xi̇̈ xj̇̈ xį̇̈ xi̇̀ xi̇́ xi̇̃ xi xi̇̈ xj xj̇̈ xį xį̇̈ xi̟̤",
		upper: "I Ï J J̈ Į Į̈ Ì Í Ĩ XÏ XJ̈ XĮ̈ XÌ XÍ XĨ XI XÏ XJ XJ̈ XĮ XĮ̈ XI̟̤",
	},

	8: {
		lang:  "lt",
		src:   "\u012e\u0300 \u00cc i\u0307\u0300 i\u0307\u0301 i\u0307\u0303 i\u0307\u0308 i\u0300\u0307",
		title: "\u012e\u0300 \u00cc \u00cc \u00cd \u0128 \u00cf I\u0300\u0307",
		lower: "\u012f\u0307\u0300 i\u0307\u0300 i\u0307\u0300 i\u0307\u0301 i\u0307\u0303 i\u0307\u0308 i\u0300\u0307",
		upper: "\u012e\u0300 \u00cc \u00cc \u00cd \u0128 \u00cf I\u0300\u0307",
	},

	9: {
		lang:  "nl",
		src:   "ijs IJs Ij Ijs İJ İJs aa aA 'ns 'S",
		title: "IJs IJs IJ IJs İj İjs Aa Aa 'ns 's",
	},

	// Note: this specification is not currently part of CLDR. The same holds
	// for the leading apostrophe handling for Dutch.
	// See http://unicode.org/cldr/trac/ticket/7078.
	10: {
		lang:  "af",
		src:   "wag 'n bietjie",
		title: "Wag 'n Bietjie",
		lower: "wag 'n bietjie",
		upper: "WAG 'N BIETJIE",
	},
}

func TestCaseMappings(t *testing.T) {
	for i, tt := range testCases {
		src, ok := tt.src.([]string)
		if !ok {
			src = strings.Split(tt.src.(string), " ")
		}

		for _, lang := range strings.Split(tt.lang, " ") {
			tag := language.MustParse(lang)
			testEntry := func(name string, mk func(language.Tag, options) transform.Transformer, gold interface{}) {
				c := Caser{mk(tag, tt.opts)}
				if gold != nil {
					wants, ok := gold.([]string)
					if !ok {
						wants = strings.Split(gold.(string), " ")
					}
					for j, want := range wants {
						if got := c.String(src[j]); got != want {
							t.Errorf("%d:%s:\n%s.String(%+q):\ngot  %+q;\nwant %+q", i, lang, name, src[j], got, want)
						}
					}
				}
				dst := make([]byte, 256) // big enough to hold any result
				src := []byte(strings.Join(src, " "))
				v := testtext.AllocsPerRun(20, func() {
					c.Transform(dst, src, true)
				})
				if v > 1.1 {
					t.Errorf("%d:%s:\n%s: number of allocs was %f; want 0", i, lang, name, v)
				}
			}
			testEntry("Upper", makeUpper, tt.upper)
			testEntry("Lower", makeLower, tt.lower)
			testEntry("Title", makeTitle, tt.title)
		}
	}
}

// TestAlloc tests that some mapping methods should not cause any allocation.
func TestAlloc(t *testing.T) {
	dst := make([]byte, 256) // big enough to hold any result
	src := []byte(txtNonASCII)

	for i, f := range []func() Caser{
		func() Caser { return Upper(language.Und) },
		func() Caser { return Lower(language.Und) },
		func() Caser { return Title(language.Und) },
	} {
		var c Caser
		v := testtext.AllocsPerRun(2, func() {
			c = f()
		})
		if v > 1 {
			// TODO: Right now only Upper has 1 allocation. Special-case Lower
			// and Title as well to have less allocations for the root locale.
			t.Skipf("%d:init: number of allocs was %f; want 0", i, v)
		}
		v = testtext.AllocsPerRun(2, func() {
			c.Transform(dst, src, true)
		})
		if v > 0 {
			t.Errorf("%d:transform: number of allocs was %f; want 0", i, v)
		}
	}
}

func TestShortBuffersAndOverflow(t *testing.T) {
	// minBufSize is the size of the buffer by which the casing operation in
	// this package are guaranteed to make progress.
	const minBufSize = norm.MaxSegmentSize

	for i, tt := range []struct {
		desc, src, want  string
		firstErr         error
		dstSize, srcSize int
		t                transform.Transformer
	}{{
		desc:     "und upper: short dst",
		src:      "abcdefg",
		want:     "ABCDEFG",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Upper(language.Und),
	}, {
		desc:     "und upper: short src",
		src:      "123é56",
		want:     "123É56",
		firstErr: transform.ErrShortSrc,
		dstSize:  4,
		srcSize:  4,
		t:        Upper(language.Und),
	}, {
		desc:     "und upper: no error on short",
		src:      "12",
		want:     "12",
		firstErr: nil,
		dstSize:  1,
		srcSize:  1,
		t:        Upper(language.Und),
	}, {
		desc:     "und lower: short dst",
		src:      "ABCDEFG",
		want:     "abcdefg",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Lower(language.Und),
	}, {
		desc:     "und lower: short src",
		src:      "123É56",
		want:     "123é56",
		firstErr: transform.ErrShortSrc,
		dstSize:  4,
		srcSize:  4,
		t:        Lower(language.Und),
	}, {
		desc:     "und lower: no error on short",
		src:      "12",
		want:     "12",
		firstErr: nil,
		dstSize:  1,
		srcSize:  1,
		t:        Lower(language.Und),
	}, {
		desc:    "final sigma: no error",
		src:     "ΟΣ",
		want:    "Ος",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und),
	}, {
		desc:     "final sigma: short source",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortSrc,
		dstSize:  minBufSize,
		srcSize:  10,
		t:        Title(language.Und),
	}, {
		desc:     "final sigma: short destination 1",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortDst,
		dstSize:  10,
		srcSize:  minBufSize,
		t:        Title(language.Und),
	}, {
		desc:     "final sigma: short destination 2",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortDst,
		dstSize:  9,
		srcSize:  minBufSize,
		t:        Title(language.Und),
	}, {
		desc:     "final sigma: short destination 3",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortDst,
		dstSize:  8,
		srcSize:  minBufSize,
		t:        Title(language.Und),
	}, {
		desc:     "clipped UTF-8 rune",
		src:      "σσσσσσσσσσσ",
		want:     "Σσσσσσσσσσσ",
		firstErr: transform.ErrShortSrc,
		dstSize:  minBufSize,
		srcSize:  5,
		t:        Title(language.Und),
	}, {
		desc:    "clipped UTF-8 rune atEOF",
		src:     "σσσ" + string([]byte{0xC0}),
		want:    "Σσσ" + string([]byte{0xC0}),
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc:    "final sigma: max ignorables",
		src:     "ΟΣ" + strings.Repeat(".", maxIgnorable) + "a",
		want:    "Οσ" + strings.Repeat(".", maxIgnorable) + "a",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc:    "final sigma: too many ignorables",
		src:     "ΟΣ" + strings.Repeat(".", maxIgnorable+1) + "a",
		want:    "Ος" + strings.Repeat(".", maxIgnorable+1) + "a",
		dstSize: minBufSize,
		srcSize: len("ΟΣ" + strings.Repeat(".", maxIgnorable+1)),
		t:       Title(language.Und),
	}, {
		desc:    "el upper: max ignorables",
		src:     "ο" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0313",
		want:    "Ο" + strings.Repeat("\u0321", maxIgnorable-1),
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Upper(language.Greek),
	}, {
		desc:    "el upper: too many ignorables",
		src:     "ο" + strings.Repeat("\u0321", maxIgnorable) + "\u0313",
		want:    "Ο" + strings.Repeat("\u0321", maxIgnorable) + "\u0313",
		dstSize: minBufSize,
		srcSize: len("ο" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Upper(language.Greek),
	}, {
		desc:     "el upper: short dst",
		src:      "123ο",
		want:     "123Ο",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Upper(language.Greek),
	}, {
		desc:    "lt lower: max ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0300",
		want:    "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Lower(language.Lithuanian),
	}, {
		desc:    "lt lower: too many ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable) + "\u0300",
		want:    "i" + strings.Repeat("\u0321", maxIgnorable) + "\u0300",
		dstSize: minBufSize,
		srcSize: len("I" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Lower(language.Lithuanian),
	}, {
		desc:     "lt lower: decomposition with short dst buffer 1",
		src:      "aaaaa\u00cc", // U+00CC LATIN CAPITAL LETTER I GRAVE
		firstErr: transform.ErrShortDst,
		want:     "aaaaai\u0307\u0300",
		dstSize:  5,
		srcSize:  minBufSize,
		t:        Lower(language.Lithuanian),
	}, {
		desc:     "lt lower: decomposition with short dst buffer 2",
		src:      "aaaa\u00cc", // U+00CC LATIN CAPITAL LETTER I GRAVE
		firstErr: transform.ErrShortDst,
		want:     "aaaai\u0307\u0300",
		dstSize:  5,
		srcSize:  minBufSize,
		t:        Lower(language.Lithuanian),
	}, {
		desc:    "lt upper: max ignorables",
		src:     "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		want:    "I" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0300",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Upper(language.Lithuanian),
	}, {
		desc:    "lt upper: too many ignorables",
		src:     "i" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		want:    "I" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		dstSize: minBufSize,
		srcSize: len("i" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Upper(language.Lithuanian),
	}, {
		desc:     "lt upper: short dst",
		src:      "12i\u0307\u0300",
		want:     "12\u00cc",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Upper(language.Lithuanian),
	}, {
		desc:    "aztr lower: max ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		want:    "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0300",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Lower(language.Turkish),
	}, {
		desc:    "aztr lower: too many ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		want:    "\u0131" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		dstSize: minBufSize,
		srcSize: len("I" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Lower(language.Turkish),
	}, {
		desc:     "nl title: pre-IJ cutoff",
		src:      "  ij",
		want:     "  IJ",
		firstErr: transform.ErrShortDst,
		dstSize:  2,
		srcSize:  minBufSize,
		t:        Title(language.Dutch),
	}, {
		desc:     "nl title: mid-IJ cutoff",
		src:      "  ij",
		want:     "  IJ",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Title(language.Dutch),
	}} {
		buf := make([]byte, tt.dstSize)
		got := []byte{}
		var nSrc, nDst int
		var err error
		for p := 0; p < len(tt.src); p += nSrc {
			q := p + tt.srcSize
			if q > len(tt.src) {
				q = len(tt.src)
			}
			nDst, nSrc, err = tt.t.Transform(buf, []byte(tt.src[p:q]), q == len(tt.src))
			got = append(got, buf[:nDst]...)

			if p == 0 && err != tt.firstErr {
				t.Errorf("%d:%s:\n error was %v; want %v", i, tt.desc, err, tt.firstErr)
				break
			}
		}
		if string(got) != tt.want {
			t.Errorf("%d:%s:\ngot  %+q;\nwant %+q", i, tt.desc, got, tt.want)
		}
	}
}

var txtASCII = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 50)

// Taken from http://creativecommons.org/licenses/by-sa/3.0/vn/
const txt_vn = `Với các điều kiện sau: Ghi nhận công của tác giả.  Nếu bạn sử
dụng, chuyển đổi, hoặc xây dựng dự án từ  nội dung được chia sẻ này, bạn phải áp
dụng giấy phép này hoặc  một giấy phép khác có các điều khoản tương tự như giấy
phép này cho dự án của bạn. Hiểu rằng: Miễn — Bất kỳ các điều kiện nào trên đây
cũng có thể được miễn bỏ nếu bạn được sự cho phép của người sở hữu bản quyền.
Phạm vi công chúng — Khi tác phẩm hoặc bất kỳ chương nào của tác phẩm đã trong
vùng dành cho công chúng theo quy định của pháp luật thì tình trạng của nó không
bị ảnh hưởng bởi giấy phép trong bất kỳ trường hợp nào.`

// http://creativecommons.org/licenses/by-sa/2.5/cn/
const txt_cn = `您可以自由： 复制、发行、展览、表演、放映、
广播或通过信息网络传播本作品 创作演绎作品
对本作品进行商业性使用 惟须遵守下列条件：
署名 — 您必须按照作者或者许可人指定的方式对作品进行署名。
相同方式共享 — 如果您改变、转换本作品或者以本作品为基础进行创作，
您只能采用与本协议相同的许可协议发布基于本作品的演绎作品。`

// Taken from http://creativecommons.org/licenses/by-sa/1.0/deed.ru
const txt_ru = `При обязательном соблюдении следующих условий: Attribution — Вы
должны атрибутировать произведение (указывать автора и источник) в порядке,
предусмотренном автором или лицензиаром (но только так, чтобы никоим образом не
подразумевалось, что они поддерживают вас или использование вами данного
произведения). Υπό τις ακόλουθες προϋποθέσεις:`

// Taken from http://creativecommons.org/licenses/by-sa/3.0/gr/
const txt_gr = `Αναφορά Δημιουργού — Θα πρέπει να κάνετε την αναφορά στο έργο με
τον τρόπο που έχει οριστεί από το δημιουργό ή το χορηγούντο την άδεια (χωρίς
όμως να εννοείται με οποιονδήποτε τρόπο ότι εγκρίνουν εσάς ή τη χρήση του έργου
από εσάς). Παρόμοια Διανομή — Εάν αλλοιώσετε, τροποποιήσετε ή δημιουργήσετε
περαιτέρω βασισμένοι στο έργο θα μπορείτε να διανέμετε το έργο που θα προκύψει
μόνο με την ίδια ή παρόμοια άδεια.`

const txtNonASCII = txt_vn + txt_cn + txt_ru + txt_gr

// TODO: Improve ASCII performance.

func benchFunc(b *testing.B, f func(b []byte) []byte, s string) {
	src := []byte(s)
	b.SetBytes(int64(len(src)))
	for i := 0; i < b.N; i++ {
		f(src)
	}
}

func benchTransformer(b *testing.B, t transform.Transformer, s string) {
	src := []byte(s)
	dst := make([]byte, len(src))
	b.SetBytes(int64(len(src)))

	for i := 0; i < b.N; i++ {
		t.Reset()
		t.Transform(dst, src, true)
	}
}

var (
	noSigma = options{noFinalSigma: true}
)

func BenchmarkBytesToLower(b *testing.B) {
	benchFunc(b, bytes.ToLower, txtNonASCII)
}

func BenchmarkSigmaLower(b *testing.B) {
	benchTransformer(b, makeLower(language.Und, options{}), txtNonASCII)
}

func BenchmarkSimpleLower(b *testing.B) {
	benchTransformer(b, makeLower(language.Und, noSigma), txtNonASCII)
}

func BenchmarkBytesToLowerASCII(b *testing.B) {
	benchFunc(b, bytes.ToLower, txtASCII)
}

func BenchmarkSigmaLowerASCII(b *testing.B) {
	benchTransformer(b, makeLower(language.Und, options{}), txtASCII)
}

func BenchmarkSimpleLowerASCII(b *testing.B) {
	benchTransformer(b, makeLower(language.Und, noSigma), txtASCII)
}

func BenchmarkBytesToTitle(b *testing.B) {
	benchFunc(b, bytes.ToTitle, txtNonASCII)
}

func BenchmarkSigmaTitle(b *testing.B) {
	benchTransformer(b, makeTitle(language.Und, options{}), txtNonASCII)
}

func BenchmarkSimpleTitle(b *testing.B) {
	benchTransformer(b, makeTitle(language.Und, noSigma), txtNonASCII)
}

func BenchmarkBytesToTitleASCII(b *testing.B) {
	benchFunc(b, bytes.ToTitle, txtASCII)
}

func BenchmarkSigmaTitleASCII(b *testing.B) {
	benchTransformer(b, makeTitle(language.Und, options{}), txtASCII)
}

func BenchmarkSimpleTitleASCII(b *testing.B) {
	benchTransformer(b, makeTitle(language.Und, noSigma), txtASCII)
}

func BenchmarkBytesUpper(b *testing.B) {
	benchFunc(b, bytes.ToUpper, txtNonASCII)
}

func BenchmarkUpper(b *testing.B) {
	benchTransformer(b, Upper(language.Und), txtNonASCII)
}

func BenchmarkBytesUpperASCII(b *testing.B) {
	benchFunc(b, bytes.ToUpper, txtASCII)
}

func BenchmarkUpperASCII(b *testing.B) {
	benchTransformer(b, Upper(language.Und), txtASCII)
}

func BenchmarkUpperSmall(b *testing.B) {
	benchTransformer(b, Upper(language.Und), "При")
}

func BenchmarkLowerSmall(b *testing.B) {
	benchTransformer(b, Lower(language.Und), "При")
}

func BenchmarkTitleSmall(b *testing.B) {
	benchTransformer(b, Title(language.Und), "при")
}
