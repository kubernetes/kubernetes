// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plural

// FormsTable are standard hard-coded plural rules.
// The application developers and the translators need to understand them.
//
// See GNU's gettext library source code: gettext/gettext-tools/src/plural-table.c
var FormsTable = []struct {
	Lang     string
	Language string
	Value    string
}{
	{"??", "Unknown", "nplurals=1; plural=0;"},
	{"ja", "Japanese", "nplurals=1; plural=0;"},
	{"vi", "Vietnamese", "nplurals=1; plural=0;"},
	{"ko", "Korean", "nplurals=1; plural=0;"},
	{"en", "English", "nplurals=2; plural=(n != 1);"},
	{"de", "German", "nplurals=2; plural=(n != 1);"},
	{"nl", "Dutch", "nplurals=2; plural=(n != 1);"},
	{"sv", "Swedish", "nplurals=2; plural=(n != 1);"},
	{"da", "Danish", "nplurals=2; plural=(n != 1);"},
	{"no", "Norwegian", "nplurals=2; plural=(n != 1);"},
	{"nb", "Norwegian Bokmal", "nplurals=2; plural=(n != 1);"},
	{"nn", "Norwegian Nynorsk", "nplurals=2; plural=(n != 1);"},
	{"fo", "Faroese", "nplurals=2; plural=(n != 1);"},
	{"es", "Spanish", "nplurals=2; plural=(n != 1);"},
	{"pt", "Portuguese", "nplurals=2; plural=(n != 1);"},
	{"it", "Italian", "nplurals=2; plural=(n != 1);"},
	{"bg", "Bulgarian", "nplurals=2; plural=(n != 1);"},
	{"el", "Greek", "nplurals=2; plural=(n != 1);"},
	{"fi", "Finnish", "nplurals=2; plural=(n != 1);"},
	{"et", "Estonian", "nplurals=2; plural=(n != 1);"},
	{"he", "Hebrew", "nplurals=2; plural=(n != 1);"},
	{"eo", "Esperanto", "nplurals=2; plural=(n != 1);"},
	{"hu", "Hungarian", "nplurals=2; plural=(n != 1);"},
	{"tr", "Turkish", "nplurals=2; plural=(n != 1);"},
	{"pt_BR", "Brazilian", "nplurals=2; plural=(n > 1);"},
	{"fr", "French", "nplurals=2; plural=(n > 1);"},
	{"lv", "Latvian", "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n != 0 ? 1 : 2);"},
	{"ga", "Irish", "nplurals=3; plural=n==1 ? 0 : n==2 ? 1 : 2;"},
	{"ro", "Romanian", "nplurals=3; plural=n==1 ? 0 : (n==0 || (n%100 > 0 && n%100 < 20)) ? 1 : 2;"},
	{"lt", "Lithuanian", "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && (n%100<10 || n%100>=20) ? 1 : 2);"},
	{"ru", "Russian", "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);"},
	{"uk", "Ukrainian", "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);"},
	{"be", "Belarusian", "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);"},
	{"sr", "Serbian", "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);"},
	{"hr", "Croatian", "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);"},
	{"cs", "Czech", "nplurals=3; plural=(n==1) ? 0 : (n>=2 && n<=4) ? 1 : 2;"},
	{"sk", "Slovak", "nplurals=3; plural=(n==1) ? 0 : (n>=2 && n<=4) ? 1 : 2;"},
	{"pl", "Polish", "nplurals=3; plural=(n==1 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);"},
	{"sl", "Slovenian", "nplurals=4; plural=(n%100==1 ? 0 : n%100==2 ? 1 : n%100==3 || n%100==4 ? 2 : 3);"},
}
