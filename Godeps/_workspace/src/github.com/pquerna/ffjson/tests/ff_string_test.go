/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package tff

import (
	"testing"
)

// Test data from https://github.com/akheron/jansson/tree/master/test/suites/valid
// jansson, Copyright (c) 2009-2014 Petri Lehtinen <petri@digip.org>
// (MIT Licensed)

func TestString(t *testing.T) {
	testType(t, &Tstring{}, &Xstring{})
	testType(t, &Tmystring{}, &Xmystring{})
	testType(t, &TmystringPtr{}, &XmystringPtr{})
}

func TestMapStringString(t *testing.T) {
	m := map[string]string{"陫ʋsş\")珷<ºɖgȏ哙ȍ": "2ħ籦ö嗏ʑ>季"}
	testCycle(t, &TMapStringString{X: m}, &XMapStringString{X: m})
}

func TestMapStringStringLong(t *testing.T) {
	m := map[string]string{"ɥ³ƞsɁ8^ʥǔTĪȸŹă": "ɩÅ議Ǹ轺@)蓳嗘TʡȂ", "丯Ƙ枛牐ɺ皚|": "\\p[", "ȉ": "ģ毋Ó6ǳ娝嘚", "ʒUɦOŖ": "斎AO6ĴC浔Ű壝ž", "/C龷ȪÆl殛瓷雼浢Ü礽绅": "D¡", "Lɋ聻鎥ʟ<$洅ɹ7\\弌Þ帺萸Do©": "A", "yǠ/淹\\韲翁&ʢsɜ": "`诫z徃鷢6ȥ啕禗Ǐ2啗塧ȱ蓿彭聡A", "瓧嫭塓烀罁胾^拜": "ǒɿʒ刽ŉ掏1ſ盷褎weǇ", "姥呄鐊唊飙Ş-U圴÷a/ɔ}摁(": "瓘ǓvjĜ蛶78Ȋ²@H", "Ĳ斬³;": "鯿r", "勽Ƙq/Ź u衲": "ŭǲ鯰硰{舁", "枊a8衍`Ĩɘ.蘯6ċV夸eɑeʤ脽ě": "6/ʕVŚ(ĿȊ甞谐颋ǅSǡƏS$+", "1ØœȠƬQg鄠": "军g>郵[+扴ȨŮ+朷Ǝ膯ǉ", "礶惇¸t颟.鵫ǚ灄鸫rʤî萨z": "", "ȶ网棊ʢ=wǕɳɷ9Ì": "'WKw(ğ儴Ůĺ}潷ʒ胵輓Ɔ", "}ȧ外ĺ稥氹Ç|¶鎚¡ Ɠ(嘒ėf倐": "窮秳ķ蟒苾h^", "?瞲Ť倱<įXŋ朘瑥A徙": "nh0åȂ町恰ǌ揠8ǉ黳鈫ʕ禒", "丩ŽoǠŻʘY賃ɪ鐊": "ľǎɳ,ǿ飏騀呣ǎ", "ȇe媹Hǝ呮}臷Ľð»ųKĵ": "踪鄌eÞȦY籎顒ǥŴ唼Ģ猇õǶț", "偐ę腬瓷碑=ɉ鎷卩蝾H韹寬娬ï瓼猀2": "ǰ溟ɴ扵閝ȝ鐵儣廡ɑ龫`劳", "ʮ馜ü": "", "șƶ4ĩĉş蝿ɖȃ賲鐅臬dH巧": "_瀹鞎sn芞QÄȻȊ+?", "E@Ȗs«ö": "蚛隖<ǶĬ4y£軶ǃ*ʙ嫙&蒒5靇C'", "忄*齧獚敆Ȏ": "螩B", "圠=l畣潁谯耨V6&]鴍Ɋ恧ȭ%ƎÜ": "涽託仭w-檮", "ʌ鴜": "琔n宂¬轚9Ȏ瀮昃2Ō¾\\", "ƅTG": "ǺƶȤ^}穠C]躢|)黰eȪ嵛4$%Q", "ǹ_Áȉ彂Ŵ廷s": "", "t莭琽§ć\\ ïì": "", "擓ƖHVe熼'FD剂讼ɓȌʟni酛": "/ɸɎ R§耶FfBls3!", "狞夌碕ʂɭ": "Ƽ@hDrȮO励鹗塢", "ʁgɸ=ǤÆ": "?讦ĭÐ", "陫ʋsş\")珷<ºɖgȏ哙ȍ": "2ħ籦ö嗏ʑ>季", "": "昕Ĭ", "Ⱦǳ@ùƸʋŀ": "ǐƲE'iþŹʣy豎@ɀ羭,铻OŤǢʭ", ">犵殇ŕ-Ɂ圯W:ĸ輦唊#v铿ʩȂ4": "屡ʁ", "1Rƥ贫d飼$俊跾|@?鷅bȻN": "H炮掊°nʮ閼咎櫸eʔŊƞ究:ho", "ƻ悖ȩ0Ƹ[": "Ndǂ>5姣>懔%熷谟þ蛯ɰ", "ŵw^Ü郀叚Fi皬择": ":5塋訩塶\"=y钡n)İ笓", "'容": "誒j剐", "猤痈C*ĕ": "鴈o_鹈ɹ坼É/pȿŘ阌"}
	testCycle(t, &TMapStringString{X: m}, &XMapStringString{X: m})
}

func TestStringEscapedControlCharacter(t *testing.T) {
	testExpectedXVal(t,
		"\x12 escaped control character",
		`\u0012 escaped control character`,
		&Xstring{})
}

func TestStringOneByteUTF8(t *testing.T) {
	testExpectedXVal(t,
		", one-byte UTF-8",
		`\u002c one-byte UTF-8`,
		&Xstring{})
}

func TestStringUtf8Escape(t *testing.T) {
	testExpectedXVal(t,
		"2ħ籦ö嗏ʑ>嫀",
		`2ħ籦ö嗏ʑ\u003e嫀`,
		&Xstring{})
}

func TestStringTwoByteUTF8(t *testing.T) {
	testExpectedXVal(t,
		"ģ two-byte UTF-8",
		`\u0123 two-byte UTF-8`,
		&Xstring{})
}

func TestStringThreeByteUTF8(t *testing.T) {
	testExpectedXVal(t,
		"ࠡ three-byte UTF-8",
		`\u0821 three-byte UTF-8`,
		&Xstring{})
}

func TestStringEsccapes(t *testing.T) {
	testExpectedXVal(t,
		`"\`+"\b\f\n\r\t",
		`\"\\\b\f\n\r\t`,
		&Xstring{})

	testExpectedXVal(t,
		`/`,
		`\/`,
		&Xstring{})
}

func TestStringSomeUTF8(t *testing.T) {
	testExpectedXVal(t,
		`€þıœəßð some utf-8 ĸʒ×ŋµåäö𝄞`,
		`€þıœəßð some utf-8 ĸʒ×ŋµåäö𝄞`,
		&Xstring{})
}

func TestBytesInString(t *testing.T) {
	testExpectedXVal(t,
		string('\xff')+` <- xFF byte`,
		string('\xff')+` <- xFF byte`,
		&Xstring{})
}

func TestString4ByteSurrogate(t *testing.T) {
	testExpectedXVal(t,
		"𝄞 surrogate, four-byte UTF-8",
		`\uD834\uDD1E surrogate, four-byte UTF-8`,
		&Xstring{})
}

func TestStringNull(t *testing.T) {
	testExpectedXValBare(t,
		"foobar",
		`null`,
		&Xstring{X: "foobar"})
}
