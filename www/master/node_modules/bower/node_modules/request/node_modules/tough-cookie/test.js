/*
 * Copyright GoInstant, Inc. and other contributors. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
'use strict';
var vows = require('vows');
var assert = require('assert');
var async = require('async');

// NOTE use require("tough-cookie") in your own code:
var tough = require('./lib/cookie');
var Cookie = tough.Cookie;
var CookieJar = tough.CookieJar;


function dateVows(table) {
  var theVows = { };
  Object.keys(table).forEach(function(date) {
    var expect = table[date];
    theVows[date] = function() {
      var got = tough.parseDate(date) ? 'valid' : 'invalid';
      assert.equal(got, expect ? 'valid' : 'invalid');
    };
  });
  return { "date parsing": theVows };
}

function matchVows(func,table) {
  var theVows = {};
  table.forEach(function(item) {
    var str = item[0];
    var dom = item[1];
    var expect = item[2];
    var label = str+(expect?" matches ":" doesn't match ")+dom;
    theVows[label] = function() {
      assert.equal(func(str,dom),expect);
    };
  });
  return theVows;
}

function defaultPathVows(table) {
  var theVows = {};
  table.forEach(function(item) {
    var str = item[0];
    var expect = item[1];
    var label = str+" gives "+expect;
    theVows[label] = function() {
      assert.equal(tough.defaultPath(str),expect);
    };
  });
  return theVows;
}

var atNow = Date.now();
function at(offset) { return {now: new Date(atNow+offset)}; }

vows.describe('Cookie Jar')
.addBatch({
  "all defined": function() {
    assert.ok(Cookie);
    assert.ok(CookieJar);
  },
})
.addBatch(
  dateVows({
    "Wed, 09 Jun 2021 10:18:14 GMT": true,
    "Wed, 09 Jun 2021 22:18:14 GMT": true,
    "Tue, 18 Oct 2011 07:42:42.123 GMT": true,
    "18 Oct 2011 07:42:42 GMT": true,
    "8 Oct 2011 7:42:42 GMT": true,
    "8 Oct 2011 7:2:42 GMT": false,
    "Oct 18 2011 07:42:42 GMT": true,
    "Tue Oct 18 2011 07:05:03 GMT+0000 (GMT)": true,
    "09 Jun 2021 10:18:14 GMT": true,
    "99 Jix 3038 48:86:72 ZMT": false,
    '01 Jan 1970 00:00:00 GMT': true,
    '01 Jan 1600 00:00:00 GMT': false, // before 1601
    '01 Jan 1601 00:00:00 GMT': true,
    '10 Feb 81 13:00:00 GMT': true, // implicit year
    'Thu, 01 Jan 1970 00:00:010 GMT': true, // strange time, non-strict OK
    'Thu, 17-Apr-2014 02:12:29 GMT': true, // dashes
    'Thu, 17-Apr-2014 02:12:29 UTC': true, // dashes and UTC
  })
)
.addBatch({
  "strict date parse of Thu, 01 Jan 1970 00:00:010 GMT": {
    topic: function() {
      return tough.parseDate('Thu, 01 Jan 1970 00:00:010 GMT', true) ? true : false;
    },
    "invalid": function(date) {
      assert.equal(date,false);
    },
  }
})
.addBatch({
  "formatting": {
    "a simple cookie": {
      topic: function() {
        var c = new Cookie();
        c.key = 'a';
        c.value = 'b';
        return c;
      },
      "validates": function(c) {
        assert.ok(c.validate());
      },
      "to string": function(c) {
        assert.equal(c.toString(), 'a=b');
      },
    },
    "a cookie with spaces in the value": {
      topic: function() {
        var c = new Cookie();
        c.key = 'a';
        c.value = 'beta gamma';
        return c;
      },
      "doesn't validate": function(c) {
        assert.ok(!c.validate());
      },
      "'garbage in, garbage out'": function(c) {
        assert.equal(c.toString(), 'a=beta gamma');
      },
    },
    "with an empty value and HttpOnly": {
      topic: function() {
        var c = new Cookie();
        c.key = 'a';
        c.httpOnly = true;
        return c;
      },
      "to string": function(c) {
        assert.equal(c.toString(), 'a=; HttpOnly');
      }
    },
    "with an expiry": {
      topic: function() {
        var c = new Cookie();
        c.key = 'a';
        c.value = 'b';
        c.setExpires("Oct 18 2011 07:05:03 GMT");
        return c;
      },
      "validates": function(c) {
        assert.ok(c.validate());
      },
      "to string": function(c) {
        assert.equal(c.toString(), 'a=b; Expires=Tue, 18 Oct 2011 07:05:03 GMT');
      },
      "to short string": function(c) {
        assert.equal(c.cookieString(), 'a=b');
      },
    },
    "with a max-age": {
      topic: function() {
        var c = new Cookie();
        c.key = 'a';
        c.value = 'b';
        c.setExpires("Oct 18 2011 07:05:03 GMT");
        c.maxAge = 12345;
        return c;
      },
      "validates": function(c) {
        assert.ok(c.validate()); // mabe this one *shouldn't*?
      },
      "to string": function(c) {
        assert.equal(c.toString(), 'a=b; Expires=Tue, 18 Oct 2011 07:05:03 GMT; Max-Age=12345');
      },
    },
    "with a bunch of things": function() {
      var c = new Cookie();
      c.key = 'a';
      c.value = 'b';
      c.setExpires("Oct 18 2011 07:05:03 GMT");
      c.maxAge = 12345;
      c.domain = 'example.com';
      c.path = '/foo';
      c.secure = true;
      c.httpOnly = true;
      c.extensions = ['MyExtension'];
      assert.equal(c.toString(), 'a=b; Expires=Tue, 18 Oct 2011 07:05:03 GMT; Max-Age=12345; Domain=example.com; Path=/foo; Secure; HttpOnly; MyExtension');
    },
    "a host-only cookie": {
      topic: function() {
        var c = new Cookie();
        c.key = 'a';
        c.value = 'b';
        c.hostOnly = true;
        c.domain = 'shouldnt-stringify.example.com';
        c.path = '/should-stringify';
        return c;
      },
      "validates": function(c) {
        assert.ok(c.validate());
      },
      "to string": function(c) {
        assert.equal(c.toString(), 'a=b; Path=/should-stringify');
      },
    },
    "minutes are '10'": {
      topic: function() {
        var c = new Cookie();
        c.key = 'a';
        c.value = 'b';
        c.expires = new Date(1284113410000);
        return c;
      },
      "validates": function(c) {
        assert.ok(c.validate());
      },
      "to string": function(c) {
        var str = c.toString();
        assert.notEqual(str, 'a=b; Expires=Fri, 010 Sep 2010 010:010:010 GMT');
        assert.equal(str, 'a=b; Expires=Fri, 10 Sep 2010 10:10:10 GMT');
      },
    }
  }
})
.addBatch({
  "TTL with max-age": function() {
    var c = new Cookie();
    c.maxAge = 123;
    assert.equal(c.TTL(), 123000);
    assert.equal(c.expiryTime(new Date(9000000)), 9123000);
  },
  "TTL with zero max-age": function() {
    var c = new Cookie();
    c.key = 'a'; c.value = 'b';
    c.maxAge = 0; // should be treated as "earliest representable"
    assert.equal(c.TTL(), 0);
    assert.equal(c.expiryTime(new Date(9000000)), -Infinity);
    assert.ok(!c.validate()); // not valid, really: non-zero-digit *DIGIT
  },
  "TTL with negative max-age": function() {
    var c = new Cookie();
    c.key = 'a'; c.value = 'b';
    c.maxAge = -1; // should be treated as "earliest representable"
    assert.equal(c.TTL(), 0);
    assert.equal(c.expiryTime(new Date(9000000)), -Infinity);
    assert.ok(!c.validate()); // not valid, really: non-zero-digit *DIGIT
  },
  "TTL with max-age and expires": function() {
    var c = new Cookie();
    c.maxAge = 123;
    c.expires = new Date(Date.now()+9000);
    assert.equal(c.TTL(), 123000);
    assert.ok(c.isPersistent());
  },
  "TTL with expires": function() {
    var c = new Cookie();
    var now = Date.now();
    c.expires = new Date(now+9000);
    assert.equal(c.TTL(now), 9000);
    assert.equal(c.expiryTime(), c.expires.getTime());
  },
  "TTL with old expires": function() {
    var c = new Cookie();
    c.setExpires('17 Oct 2010 00:00:00 GMT');
    assert.ok(c.TTL() < 0);
    assert.ok(c.isPersistent());
  },
  "default TTL": {
    topic: function() { return new Cookie(); },
    "is Infinite-future": function(c) { assert.equal(c.TTL(), Infinity) },
    "is a 'session' cookie": function(c) { assert.ok(!c.isPersistent()) },
  },
}).addBatch({
  "Parsing": {
    "simple": {
      topic: function() {
        return Cookie.parse('a=bcd',true) || null;
      },
      "parsed": function(c) { assert.ok(c) },
      "key": function(c) { assert.equal(c.key, 'a') },
      "value": function(c) { assert.equal(c.value, 'bcd') },
      "no path": function(c) { assert.equal(c.path, null) },
      "no domain": function(c) { assert.equal(c.domain, null) },
      "no extensions": function(c) { assert.ok(!c.extensions) },
    },
    "with expiry": {
      topic: function() {
        return Cookie.parse('a=bcd; Expires=Tue, 18 Oct 2011 07:05:03 GMT',true) || null;
      },
      "parsed": function(c) { assert.ok(c) },
      "key": function(c) { assert.equal(c.key, 'a') },
      "value": function(c) { assert.equal(c.value, 'bcd') },
      "has expires": function(c) {
        assert.ok(c.expires !== Infinity, 'expiry is infinite when it shouldn\'t be');
        assert.equal(c.expires.getTime(), 1318921503000);
      },
    },
    "with expiry and path": {
      topic: function() {
        return Cookie.parse('abc="xyzzy!"; Expires=Tue, 18 Oct 2011 07:05:03 GMT; Path=/aBc',true) || null;
      },
      "parsed": function(c) { assert.ok(c) },
      "key": function(c) { assert.equal(c.key, 'abc') },
      "value": function(c) { assert.equal(c.value, 'xyzzy!') },
      "has expires": function(c) {
        assert.ok(c.expires !== Infinity, 'expiry is infinite when it shouldn\'t be');
        assert.equal(c.expires.getTime(), 1318921503000);
      },
      "has path": function(c) { assert.equal(c.path, '/aBc'); },
      "no httponly or secure": function(c) {
        assert.ok(!c.httpOnly);
        assert.ok(!c.secure);
      },
    },
    "with everything": {
      topic: function() {
        return Cookie.parse('abc="xyzzy!"; Expires=Tue, 18 Oct 2011 07:05:03 GMT; Path=/aBc; Domain=example.com; Secure; HTTPOnly; Max-Age=1234; Foo=Bar; Baz', true) || null;
      },
      "parsed": function(c) { assert.ok(c) },
      "key": function(c) { assert.equal(c.key, 'abc') },
      "value": function(c) { assert.equal(c.value, 'xyzzy!') },
      "has expires": function(c) {
        assert.ok(c.expires !== Infinity, 'expiry is infinite when it shouldn\'t be');
        assert.equal(c.expires.getTime(), 1318921503000);
      },
      "has path": function(c) { assert.equal(c.path, '/aBc'); },
      "has domain": function(c) { assert.equal(c.domain, 'example.com'); },
      "has httponly": function(c) { assert.equal(c.httpOnly, true); },
      "has secure": function(c) { assert.equal(c.secure, true); },
      "has max-age": function(c) { assert.equal(c.maxAge, 1234); },
      "has extensions": function(c) {
        assert.ok(c.extensions);
        assert.equal(c.extensions[0], 'Foo=Bar');
        assert.equal(c.extensions[1], 'Baz');
      },
    },
    "invalid expires": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; Expires=xyzzy", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; Expires=xyzzy");
        assert.ok(c);
        assert.equal(c.expires, Infinity);
      },
    },
    "zero max-age": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; Max-Age=0", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; Max-Age=0");
        assert.ok(c);
        assert.equal(c.maxAge, 0);
      },
    },
    "negative max-age": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; Max-Age=-1", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; Max-Age=-1");
        assert.ok(c);
        assert.equal(c.maxAge, -1);
      },
    },
    "empty domain": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; domain=", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; domain=");
        assert.ok(c);
        assert.equal(c.domain, null);
      },
    },
    "dot domain": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; domain=.", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; domain=.");
        assert.ok(c);
        assert.equal(c.domain, null);
      },
    },
    "uppercase domain": {
      "strict lowercases": function() {
        var c = Cookie.parse("a=b; domain=EXAMPLE.COM");
        assert.ok(c);
        assert.equal(c.domain, 'example.com');
      },
      "non-strict lowercases": function() {
        var c = Cookie.parse("a=b; domain=EXAMPLE.COM");
        assert.ok(c);
        assert.equal(c.domain, 'example.com');
      },
    },
    "trailing dot in domain": {
      topic: function() {
        return Cookie.parse("a=b; Domain=example.com.", true) || null;
      },
      "has the domain": function(c) { assert.equal(c.domain,"example.com.") },
      "but doesn't validate": function(c) { assert.equal(c.validate(),false) },
    },
    "empty path": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; path=", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; path=");
        assert.ok(c);
        assert.equal(c.path, null);
      },
    },
    "no-slash path": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; path=xyzzy", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; path=xyzzy");
        assert.ok(c);
        assert.equal(c.path, null);
      },
    },
    "trailing semi-colons after path": {
      topic: function () {
        return [
          "a=b; path=/;",
          "c=d;;;;"
        ];
      },
      "strict": function (t) {
        assert.ok(!Cookie.parse(t[0], true));
        assert.ok(!Cookie.parse(t[1], true));
      },
      "non-strict": function (t) {
        var c1 = Cookie.parse(t[0]);
        var c2 = Cookie.parse(t[1]);
        assert.ok(c1);
        assert.ok(c2);
        assert.equal(c1.path, '/');
      }
    },
    "secure-with-value": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; Secure=xyzzy", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; Secure=xyzzy");
        assert.ok(c);
        assert.equal(c.secure, true);
      },
    },
    "httponly-with-value": {
      "strict": function() { assert.ok(!Cookie.parse("a=b; HttpOnly=xyzzy", true)) },
      "non-strict": function() {
        var c = Cookie.parse("a=b; HttpOnly=xyzzy");
        assert.ok(c);
        assert.equal(c.httpOnly, true);
      },
    },
    "garbage": {
      topic: function() {
        return Cookie.parse("\x08", true) || null;
      },
      "doesn't parse": function(c) { assert.equal(c,null) },
    },
    "public suffix domain": {
      topic: function() {
        return Cookie.parse("a=b; domain=kyoto.jp", true) || null;
      },
      "parses fine": function(c) {
        assert.ok(c);
        assert.equal(c.domain, 'kyoto.jp');
      },
      "but fails validation": function(c) {
        assert.ok(c);
        assert.ok(!c.validate());
      },
    },
    "Ironically, Google 'GAPS' cookie has very little whitespace": {
      topic: function() {
        return Cookie.parse("GAPS=1:A1aaaaAaAAa1aaAaAaaAAAaaa1a11a:aaaAaAaAa-aaaA1-;Path=/;Expires=Thu, 17-Apr-2014 02:12:29 GMT;Secure;HttpOnly");
      },
      "parsed": function(c) { assert.ok(c) },
      "key": function(c) { assert.equal(c.key, 'GAPS') },
      "value": function(c) { assert.equal(c.value, '1:A1aaaaAaAAa1aaAaAaaAAAaaa1a11a:aaaAaAaAa-aaaA1-') },
      "path": function(c) {
        assert.notEqual(c.path, '/;Expires'); // BUG
        assert.equal(c.path, '/');
      },
      "expires": function(c) {
        assert.notEqual(c.expires, Infinity);
        assert.equal(c.expires.getTime(), 1397700749000);
      },
      "secure": function(c) { assert.ok(c.secure) },
      "httponly": function(c) { assert.ok(c.httpOnly) },
    },
    "lots of equal signs": {
      topic: function() {
        return Cookie.parse("queryPref=b=c&d=e; Path=/f=g; Expires=Thu, 17 Apr 2014 02:12:29 GMT; HttpOnly");
      },
      "parsed": function(c) { assert.ok(c) },
      "key": function(c) { assert.equal(c.key, 'queryPref') },
      "value": function(c) { assert.equal(c.value, 'b=c&d=e') },
      "path": function(c) {
        assert.equal(c.path, '/f=g');
      },
      "expires": function(c) {
        assert.notEqual(c.expires, Infinity);
        assert.equal(c.expires.getTime(), 1397700749000);
      },
      "httponly": function(c) { assert.ok(c.httpOnly) },
    },
    "spaces in value": {
      "strict": {
        topic: function() {
          return Cookie.parse('a=one two three',true) || null;
        },
        "did not parse": function(c) { assert.isNull(c) },
      },
      "non-strict": {
        topic: function() {
          return Cookie.parse('a=one two three',false) || null;
        },
        "parsed": function(c) { assert.ok(c) },
        "key": function(c) { assert.equal(c.key, 'a') },
        "value": function(c) { assert.equal(c.value, 'one two three') },
        "no path": function(c) { assert.equal(c.path, null) },
        "no domain": function(c) { assert.equal(c.domain, null) },
        "no extensions": function(c) { assert.ok(!c.extensions) },
      },
    },
    "quoted spaces in value": {
      "strict": {
        topic: function() {
          return Cookie.parse('a="one two three"',true) || null;
        },
        "did not parse": function(c) { assert.isNull(c) },
      },
      "non-strict": {
        topic: function() {
          return Cookie.parse('a="one two three"',false) || null;
        },
        "parsed": function(c) { assert.ok(c) },
        "key": function(c) { assert.equal(c.key, 'a') },
        "value": function(c) { assert.equal(c.value, 'one two three') },
        "no path": function(c) { assert.equal(c.path, null) },
        "no domain": function(c) { assert.equal(c.domain, null) },
        "no extensions": function(c) { assert.ok(!c.extensions) },
      }
    },
    "non-ASCII in value": {
      "strict": {
        topic: function() {
          return Cookie.parse('farbe=weiß',true) || null;
        },
        "did not parse": function(c) { assert.isNull(c) },
      },
      "non-strict": {
        topic: function() {
          return Cookie.parse('farbe=weiß',false) || null;
        },
        "parsed": function(c) { assert.ok(c) },
        "key": function(c) { assert.equal(c.key, 'farbe') },
        "value": function(c) { assert.equal(c.value, 'weiß') },
        "no path": function(c) { assert.equal(c.path, null) },
        "no domain": function(c) { assert.equal(c.domain, null) },
        "no extensions": function(c) { assert.ok(!c.extensions) },
      },
    },
  }
})
.addBatch({
  "domain normalization": {
    "simple": function() {
      var c = new Cookie();
      c.domain = "EXAMPLE.com";
      assert.equal(c.canonicalizedDomain(), "example.com");
    },
    "extra dots": function() {
      var c = new Cookie();
      c.domain = ".EXAMPLE.com";
      assert.equal(c.cdomain(), "example.com");
    },
    "weird trailing dot": function() {
      var c = new Cookie();
      c.domain = "EXAMPLE.ca.";
      assert.equal(c.canonicalizedDomain(), "example.ca.");
    },
    "weird internal dots": function() {
      var c = new Cookie();
      c.domain = "EXAMPLE...ca.";
      assert.equal(c.canonicalizedDomain(), "example...ca.");
    },
    "IDN": function() {
      var c = new Cookie();
      c.domain = "δοκιμή.δοκιμή"; // "test.test" in greek
      assert.equal(c.canonicalizedDomain(), "xn--jxalpdlp.xn--jxalpdlp");
    }
  }
})
.addBatch({
  "Domain Match":matchVows(tough.domainMatch, [
    // str,          dom,          expect
    ["example.com", "example.com", true],
    ["eXaMpLe.cOm", "ExAmPlE.CoM", true],
    ["no.ca", "yes.ca", false],
    ["wwwexample.com", "example.com", false],
    ["www.example.com", "example.com", true],
    ["example.com", "www.example.com", false],
    ["www.subdom.example.com", "example.com", true],
    ["www.subdom.example.com", "subdom.example.com", true],
    ["example.com", "example.com.", false], // RFC6265 S4.1.2.3
    ["192.168.0.1", "168.0.1", false], // S5.1.3 "The string is a host name"
    [null, "example.com", null],
    ["example.com", null, null],
    [null, null, null],
    [undefined, undefined, null],
  ])
})
.addBatch({
  "default-path": defaultPathVows([
    [null,"/"],
    ["/","/"],
    ["/file","/"],
    ["/dir/file","/dir"],
    ["noslash","/"],
  ])
})
.addBatch({
  "Path-Match": matchVows(tough.pathMatch, [
    // request, cookie, match
    ["/","/",true],
    ["/dir","/",true],
    ["/","/dir",false],
    ["/dir/","/dir/", true],
    ["/dir/file","/dir/",true],
    ["/dir/file","/dir",true],
    ["/directory","/dir",false],
  ])
})
.addBatch({
  "Cookie Sorting": {
    topic: function() {
      var cookies = [];
      var now = Date.now();
      cookies.push(Cookie.parse("a=0; Domain=example.com"));
      cookies.push(Cookie.parse("b=1; Domain=www.example.com"));
      cookies.push(Cookie.parse("c=2; Domain=example.com; Path=/pathA"));
      cookies.push(Cookie.parse("d=3; Domain=www.example.com; Path=/pathA"));
      cookies.push(Cookie.parse("e=4; Domain=example.com; Path=/pathA/pathB"));
      cookies.push(Cookie.parse("f=5; Domain=www.example.com; Path=/pathA/pathB"));

      // force a stable creation time consistent with the order above since
      // some may have been created at now + 1ms.
      var i = cookies.length;
      cookies.forEach(function(cookie) {
        cookie.creation = new Date(now - 100*(i--));
      });

      // weak shuffle:
      cookies = cookies.sort(function(){return Math.random()-0.5});

      cookies = cookies.sort(tough.cookieCompare);
      return cookies;
    },
    "got": function(cookies) {
      assert.lengthOf(cookies, 6);
      var names = cookies.map(function(c) {return c.key});
      assert.deepEqual(names, ['e','f','c','d','a','b']);
    },
  }
})
.addBatch({
  "CookieJar": {
    "Setting a basic cookie": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=example.com; Path=/");
        assert.strictEqual(c.hostOnly, null);
        assert.instanceOf(c.creation, Date);
        assert.strictEqual(c.lastAccessed, null);
        c.creation = new Date(Date.now()-10000);
        cj.setCookie(c, 'http://example.com/index.html', this.callback);
      },
      "works": function(c) { assert.instanceOf(c,Cookie) }, // C is for Cookie, good enough for me
      "gets timestamped": function(c) {
        assert.ok(c.creation);
        assert.ok(Date.now() - c.creation.getTime() < 5000); // recently stamped
        assert.ok(c.lastAccessed);
        assert.equal(c.creation, c.lastAccessed);
        assert.equal(c.TTL(), Infinity);
        assert.ok(!c.isPersistent());
      },
    },
    "Setting a no-path cookie": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=example.com");
        assert.strictEqual(c.hostOnly, null);
        assert.instanceOf(c.creation, Date);
        assert.strictEqual(c.lastAccessed, null);
        c.creation = new Date(Date.now()-10000);
        cj.setCookie(c, 'http://example.com/index.html', this.callback);
      },
      "domain": function(c) { assert.equal(c.domain, 'example.com') },
      "path is /": function(c) { assert.equal(c.path, '/') },
      "path was derived": function(c) { assert.strictEqual(c.pathIsDefault, true) },
    },
    "Setting a cookie already marked as host-only": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=example.com");
        assert.strictEqual(c.hostOnly, null);
        assert.instanceOf(c.creation, Date);
        assert.strictEqual(c.lastAccessed, null);
        c.creation = new Date(Date.now()-10000);
        c.hostOnly = true;
        cj.setCookie(c, 'http://example.com/index.html', this.callback);
      },
      "domain": function(c) { assert.equal(c.domain, 'example.com') },
      "still hostOnly": function(c) { assert.strictEqual(c.hostOnly, true) },
    },
    "Setting a session cookie": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b");
        assert.strictEqual(c.path, null);
        cj.setCookie(c, 'http://www.example.com/dir/index.html', this.callback);
      },
      "works": function(c) { assert.instanceOf(c,Cookie) },
      "gets the domain": function(c) { assert.equal(c.domain, 'www.example.com') },
      "gets the default path": function(c) { assert.equal(c.path, '/dir') },
      "is 'hostOnly'": function(c) { assert.ok(c.hostOnly) },
    },
    "Setting wrong domain cookie": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=fooxample.com; Path=/");
        cj.setCookie(c, 'http://example.com/index.html', this.callback);
      },
      "fails": function(err,c) {
        assert.ok(err.message.match(/domain/i));
        assert.ok(!c);
      },
    },
    "Setting sub-domain cookie": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=www.example.com; Path=/");
        cj.setCookie(c, 'http://example.com/index.html', this.callback);
      },
      "fails": function(err,c) {
        assert.ok(err.message.match(/domain/i));
        assert.ok(!c);
      },
    },
    "Setting super-domain cookie": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=example.com; Path=/");
        cj.setCookie(c, 'http://www.app.example.com/index.html', this.callback);
      },
      "success": function(err,c) {
        assert.ok(!err);
        assert.equal(c.domain, 'example.com');
      },
    },
    "Setting a sub-path cookie on a super-domain": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=example.com; Path=/subpath");
        assert.strictEqual(c.hostOnly, null);
        assert.instanceOf(c.creation, Date);
        assert.strictEqual(c.lastAccessed, null);
        c.creation = new Date(Date.now()-10000);
        cj.setCookie(c, 'http://www.example.com/index.html', this.callback);
      },
      "domain is super-domain": function(c) { assert.equal(c.domain, 'example.com') },
      "path is /subpath": function(c) { assert.equal(c.path, '/subpath') },
      "path was NOT derived": function(c) { assert.strictEqual(c.pathIsDefault, null) },
    },
    "Setting HttpOnly cookie over non-HTTP API": {
      topic: function() {
        var cj = new CookieJar();
        var c = Cookie.parse("a=b; Domain=example.com; Path=/; HttpOnly");
        cj.setCookie(c, 'http://example.com/index.html', {http:false}, this.callback);
      },
      "fails": function(err,c) {
        assert.match(err.message, /HttpOnly/i);
        assert.ok(!c);
      },
    },
  },
  "Cookie Jar store eight cookies": {
    topic: function() {
      var cj = new CookieJar();
      var ex = 'http://example.com/index.html';
      var tasks = [];
      tasks.push(function(next) {
        cj.setCookie('a=1; Domain=example.com; Path=/',ex,at(0),next);
      });
      tasks.push(function(next) {
        cj.setCookie('b=2; Domain=example.com; Path=/; HttpOnly',ex,at(1000),next);
      });
      tasks.push(function(next) {
        cj.setCookie('c=3; Domain=example.com; Path=/; Secure',ex,at(2000),next);
      });
      tasks.push(function(next) { // path
        cj.setCookie('d=4; Domain=example.com; Path=/foo',ex,at(3000),next);
      });
      tasks.push(function(next) { // host only
        cj.setCookie('e=5',ex,at(4000),next);
      });
      tasks.push(function(next) { // other domain
        cj.setCookie('f=6; Domain=nodejs.org; Path=/','http://nodejs.org',at(5000),next);
      });
      tasks.push(function(next) { // expired
        cj.setCookie('g=7; Domain=example.com; Path=/; Expires=Tue, 18 Oct 2011 00:00:00 GMT',ex,at(6000),next);
      });
      tasks.push(function(next) { // expired via Max-Age
        cj.setCookie('h=8; Domain=example.com; Path=/; Max-Age=1',ex,next);
      });
      var cb = this.callback;
      async.parallel(tasks, function(err,results){
        setTimeout(function() {
          cb(err,cj,results);
        }, 2000); // so that 'h=8' expires
      });
    },
    "setup ok": function(err,cj,results) {
      assert.ok(!err);
      assert.ok(cj);
      assert.ok(results);
    },
    "then retrieving for http://nodejs.org": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('http://nodejs.org',this.callback);
      },
      "get a nodejs cookie": function(cookies) {
        assert.lengthOf(cookies, 1);
        var cookie = cookies[0];
        assert.equal(cookie.domain, 'nodejs.org');
      },
    },
    "then retrieving for https://example.com": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('https://example.com',{secure:true},this.callback);
      },
      "get a secure example cookie with others": function(cookies) {
        var names = cookies.map(function(c) {return c.key});
        assert.deepEqual(names, ['a','b','c','e']);
      },
    },
    "then retrieving for https://example.com (missing options)": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('https://example.com',this.callback);
      },
      "get a secure example cookie with others": function(cookies) {
        var names = cookies.map(function(c) {return c.key});
        assert.deepEqual(names, ['a','b','c','e']);
      },
    },
    "then retrieving for http://example.com": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('http://example.com',this.callback);
      },
      "get a bunch of cookies": function(cookies) {
        var names = cookies.map(function(c) {return c.key});
        assert.deepEqual(names, ['a','b','e']);
      },
    },
    "then retrieving for http://EXAMPlE.com": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('http://EXAMPlE.com',this.callback);
      },
      "get a bunch of cookies": function(cookies) {
        var names = cookies.map(function(c) {return c.key});
        assert.deepEqual(names, ['a','b','e']);
      },
    },
    "then retrieving for http://example.com, non-HTTP": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('http://example.com',{http:false},this.callback);
      },
      "get a bunch of cookies": function(cookies) {
        var names = cookies.map(function(c) {return c.key});
        assert.deepEqual(names, ['a','e']);
      },
    },
    "then retrieving for http://example.com/foo/bar": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('http://example.com/foo/bar',this.callback);
      },
      "get a bunch of cookies": function(cookies) {
        var names = cookies.map(function(c) {return c.key});
        assert.deepEqual(names, ['d','a','b','e']);
      },
    },
    "then retrieving for http://example.com as a string": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookieString('http://example.com',this.callback);
      },
      "get a single string": function(cookieHeader) {
        assert.equal(cookieHeader, "a=1; b=2; e=5");
      },
    },
    "then retrieving for http://example.com as a set-cookie header": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getSetCookieStrings('http://example.com',this.callback);
      },
      "get a single string": function(cookieHeaders) {
        assert.lengthOf(cookieHeaders, 3);
        assert.equal(cookieHeaders[0], "a=1; Domain=example.com; Path=/");
        assert.equal(cookieHeaders[1], "b=2; Domain=example.com; Path=/; HttpOnly");
        assert.equal(cookieHeaders[2], "e=5; Path=/");
      },
    },
    "then retrieving for http://www.example.com/": {
      topic: function(cj,oldResults) {
        assert.ok(oldResults);
        cj.getCookies('http://www.example.com/foo/bar',this.callback);
      },
      "get a bunch of cookies": function(cookies) {
        var names = cookies.map(function(c) {return c.key});
        assert.deepEqual(names, ['d','a','b']); // note lack of 'e'
      },
    },
  },
  "Repeated names": {
    topic: function() {
      var cb = this.callback;
      var cj = new CookieJar();
      var ex = 'http://www.example.com/';
      var sc = cj.setCookie;
      var tasks = [];
      var now = Date.now();
      tasks.push(sc.bind(cj,'aaaa=xxxx',ex,at(0)));
      tasks.push(sc.bind(cj,'aaaa=1111; Domain=www.example.com',ex,at(1000)));
      tasks.push(sc.bind(cj,'aaaa=2222; Domain=example.com',ex,at(2000)));
      tasks.push(sc.bind(cj,'aaaa=3333; Domain=www.example.com; Path=/pathA',ex,at(3000)));
      async.series(tasks,function(err,results) {
        results = results.filter(function(e) {return e !== undefined});
        cb(err,{cj:cj, cookies:results, now:now});
      });
    },
    "all got set": function(err,t) {
      assert.lengthOf(t.cookies,4);
    },
    "then getting 'em back": {
      topic: function(t) {
        var cj = t.cj;
        cj.getCookies('http://www.example.com/pathA',this.callback);
      },
      "there's just three": function (err,cookies) {
        var vals = cookies.map(function(c) {return c.value});
        // may break with sorting; sorting should put 3333 first due to longest path:
        assert.deepEqual(vals, ['3333','1111','2222']);
      }
    },
  },
  "CookieJar setCookie errors": {
    "public-suffix domain": {
      topic: function() {
        var cj = new CookieJar();
        cj.setCookie('i=9; Domain=kyoto.jp; Path=/','kyoto.jp',this.callback);
      },
      "errors": function(err,cookie) {
        assert.ok(err);
        assert.ok(!cookie);
        assert.match(err.message, /public suffix/i);
      },
    },
    "wrong domain": {
      topic: function() {
        var cj = new CookieJar();
        cj.setCookie('j=10; Domain=google.com; Path=/','google.ca',this.callback);
      },
      "errors": function(err,cookie) {
        assert.ok(err);
        assert.ok(!cookie);
        assert.match(err.message, /not in this host's domain/i);
      },
    },
    "old cookie is HttpOnly": {
      topic: function() {
        var cb = this.callback;
        var next = function (err,c) {
          c = null;
          return cb(err,cj);
        };
        var cj = new CookieJar();
        cj.setCookie('k=11; Domain=example.ca; Path=/; HttpOnly','http://example.ca',{http:true},next);
      },
      "initial cookie is set": function(err,cj) {
        assert.ok(!err);
        assert.ok(cj);
      },
      "but when trying to overwrite": {
        topic: function(cj) {
          var cb = this.callback;
          var next = function(err,c) {
            c = null;
            cb(null,err);
          };
          cj.setCookie('k=12; Domain=example.ca; Path=/','http://example.ca',{http:false},next);
        },
        "it's an error": function(err) {
          assert.ok(err);
        },
        "then, checking the original": {
          topic: function(ignored,cj) {
            assert.ok(cj instanceof CookieJar);
            cj.getCookies('http://example.ca',{http:true},this.callback);
          },
          "cookie has original value": function(err,cookies) {
            assert.equal(err,null);
            assert.lengthOf(cookies, 1);
            assert.equal(cookies[0].value,11);
          },
        },
      },
    },
  },
})
.addBatch({
  "JSON": {
    "serialization": {
      topic: function() {
        var c = Cookie.parse('alpha=beta; Domain=example.com; Path=/foo; Expires=Tue, 19 Jan 2038 03:14:07 GMT; HttpOnly');
        return JSON.stringify(c);
      },
      "gives a string": function(str) {
        assert.equal(typeof str, "string");
      },
      "date is in ISO format": function(str) {
        assert.match(str, /"expires":"2038-01-19T03:14:07\.000Z"/, 'expires is in ISO format');
      },
    },
    "deserialization": {
      topic: function() {
        var json = '{"key":"alpha","value":"beta","domain":"example.com","path":"/foo","expires":"2038-01-19T03:14:07.000Z","httpOnly":true,"lastAccessed":2000000000123}';
        return Cookie.fromJSON(json);
      },
      "works": function(c) {
        assert.ok(c);
      },
      "key": function(c) { assert.equal(c.key, "alpha") },
      "value": function(c) { assert.equal(c.value, "beta") },
      "domain": function(c) { assert.equal(c.domain, "example.com") },
      "path": function(c) { assert.equal(c.path, "/foo") },
      "httpOnly": function(c) { assert.strictEqual(c.httpOnly, true) },
      "secure": function(c) { assert.strictEqual(c.secure, false) },
      "hostOnly": function(c) { assert.strictEqual(c.hostOnly, null) },
      "expires is a date object": function(c) {
        assert.equal(c.expires.getTime(), 2147483647000);
      },
      "lastAccessed is a date object": function(c) {
        assert.equal(c.lastAccessed.getTime(), 2000000000123);
      },
      "creation defaulted": function(c) {
        assert.ok(c.creation.getTime());
      }
    },
    "null deserialization": {
      topic: function() {
        return Cookie.fromJSON(null);
      },
      "is null": function(cookie) {
        assert.equal(cookie,null);
      },
    },
  },
  "expiry deserialization": {
    "Infinity": {
      topic: Cookie.fromJSON.bind(null, '{"expires":"Infinity"}'),
      "is infinite": function(c) {
        assert.strictEqual(c.expires, "Infinity");
        assert.equal(c.expires, Infinity);
      },
    },
  },
  "maxAge serialization": {
    topic: function() {
      return function(toSet) {
        var c = new Cookie();
        c.key = 'foo'; c.value = 'bar';
        c.setMaxAge(toSet);
        return JSON.stringify(c);
      };
    },
    "zero": {
      topic: function(f) { return f(0) },
      "looks good": function(str) {
        assert.match(str, /"maxAge":0/);
      },
    },
    "Infinity": {
      topic: function(f) { return f(Infinity) },
      "looks good": function(str) {
        assert.match(str, /"maxAge":"Infinity"/);
      },
    },
    "-Infinity": {
      topic: function(f) { return f(-Infinity) },
      "looks good": function(str) {
        assert.match(str, /"maxAge":"-Infinity"/);
      },
    },
    "null": {
      topic: function(f) { return f(null) },
      "looks good": function(str) {
        assert.match(str, /"maxAge":null/);
      },
    },
  },
  "maxAge deserialization": {
    "number": {
      topic: Cookie.fromJSON.bind(null,'{"key":"foo","value":"bar","maxAge":123}'),
      "is the number": function(c) {
        assert.strictEqual(c.maxAge, 123);
      },
    },
    "null": {
      topic: Cookie.fromJSON.bind(null,'{"key":"foo","value":"bar","maxAge":null}'),
      "is null": function(c) {
        assert.strictEqual(c.maxAge, null);
      },
    },
    "less than zero": {
      topic: Cookie.fromJSON.bind(null,'{"key":"foo","value":"bar","maxAge":-123}'),
      "is -123": function(c) {
        assert.strictEqual(c.maxAge, -123);
      },
    },
    "Infinity": {
      topic: Cookie.fromJSON.bind(null,'{"key":"foo","value":"bar","maxAge":"Infinity"}'),
      "is inf-as-string": function(c) {
        assert.strictEqual(c.maxAge, "Infinity");
      },
    },
    "-Infinity": {
      topic: Cookie.fromJSON.bind(null,'{"key":"foo","value":"bar","maxAge":"-Infinity"}'),
      "is inf-as-string": function(c) {
        assert.strictEqual(c.maxAge, "-Infinity");
      },
    },
  }
})
.addBatch({
  "permuteDomain": {
    "base case": {
      topic: tough.permuteDomain.bind(null,'example.com'),
      "got the domain": function(list) {
        assert.deepEqual(list, ['example.com']);
      },
    },
    "two levels": {
      topic: tough.permuteDomain.bind(null,'foo.bar.example.com'),
      "got three things": function(list) {
        assert.deepEqual(list, ['example.com','bar.example.com','foo.bar.example.com']);
      },
    },
    "invalid domain": {
      topic: tough.permuteDomain.bind(null,'foo.bar.example.localduhmain'),
      "got three things": function(list) {
        assert.equal(list, null);
      },
    },
  },
  "permutePath": {
    "base case": {
      topic: tough.permutePath.bind(null,'/'),
      "just slash": function(list) {
        assert.deepEqual(list,['/']);
      },
    },
    "single case": {
      topic: tough.permutePath.bind(null,'/foo'),
      "two things": function(list) {
        assert.deepEqual(list,['/foo','/']);
      },
      "path matching": function(list) {
        list.forEach(function(e) {
          assert.ok(tough.pathMatch('/foo',e));
        });
      },
    },
    "double case": {
      topic: tough.permutePath.bind(null,'/foo/bar'),
      "four things": function(list) {
        assert.deepEqual(list,['/foo/bar','/foo','/']);
      },
      "path matching": function(list) {
        list.forEach(function(e) {
          assert.ok(tough.pathMatch('/foo/bar',e));
        });
      },
    },
    "trailing slash": {
      topic: tough.permutePath.bind(null,'/foo/bar/'),
      "three things": function(list) {
        assert.deepEqual(list,['/foo/bar','/foo','/']);
      },
      "path matching": function(list) {
        list.forEach(function(e) {
          assert.ok(tough.pathMatch('/foo/bar/',e));
        });
      },
    },
  }
})
.addBatch({
  "Issue 1": {
    topic: function() {
      var cj = new CookieJar();
      cj.setCookie('hello=world; path=/some/path/', 'http://domain/some/path/file', function(err,cookie) {
        this.callback(err,{cj:cj, cookie:cookie});
      }.bind(this));
    },
    "stored a cookie": function(t) {
      assert.ok(t.cookie);
    },
    "cookie's path was modified to remove unnecessary slash": function(t) {
      assert.equal(t.cookie.path, '/some/path');
    },
    "getting it back": {
      topic: function(t) {
        t.cj.getCookies('http://domain/some/path/file', function(err,cookies) {
          this.callback(err, {cj:t.cj, cookies:cookies||[]});
        }.bind(this));
      },
      "got one cookie": function(t) {
        assert.lengthOf(t.cookies, 1);
      },
      "it's the right one": function(t) {
        var c = t.cookies[0];
        assert.equal(c.key, 'hello');
        assert.equal(c.value, 'world');
      },
    }
  }
})
.addBatch({
  "expiry option": {
    topic: function() {
      var cb = this.callback;
      var cj = new CookieJar();
      cj.setCookie('near=expiry; Domain=example.com; Path=/; Max-Age=1','http://www.example.com',at(-1), function(err,cookie) {

        cb(err, {cj:cj, cookie:cookie});
      });
    },
    "set the cookie": function(t) {
      assert.ok(t.cookie, "didn't set?!");
      assert.equal(t.cookie.key, 'near');
    },
    "then, retrieving": {
      topic: function(t) {
        var cb = this.callback;
        setTimeout(function() {
          t.cj.getCookies('http://www.example.com', {http:true, expire:false}, function(err,cookies) {
            t.cookies = cookies;
            cb(err,t);
          });
        },2000);
      },
      "got the cookie": function(t) {
        assert.lengthOf(t.cookies, 1);
        assert.equal(t.cookies[0].key, 'near');
      },
    }
  }
})
.addBatch({
  "trailing semi-colon set into cj": {
    topic: function () {
      var cb = this.callback;
      var cj = new CookieJar();
      var ex = 'http://www.example.com';
      var tasks = [];
      tasks.push(function(next) {
        cj.setCookie('broken_path=testme; path=/;',ex,at(-1),next);
      });
      tasks.push(function(next) {
        cj.setCookie('b=2; Path=/;;;;',ex,at(-1),next);
      });
      async.parallel(tasks, function (err, cookies) {
        cb(null, {
          cj: cj,
          cookies: cookies
        });
      });
    },
    "check number of cookies": function (t) {
      assert.lengthOf(t.cookies, 2, "didn't set");
    },
    "check *broken_path* was set properly": function (t) {
      assert.equal(t.cookies[0].key, "broken_path");
      assert.equal(t.cookies[0].value, "testme");
      assert.equal(t.cookies[0].path, "/");
    },
    "check *b* was set properly": function (t) {
      assert.equal(t.cookies[1].key, "b");
      assert.equal(t.cookies[1].value, "2");
      assert.equal(t.cookies[1].path, "/");
    },
    "retrieve the cookie": {
      topic: function (t) {
        var cb = this.callback;
        t.cj.getCookies('http://www.example.com', {}, function (err, cookies) {
          t.cookies = cookies;
          cb(err, t);
        });
      },
      "get the cookie": function(t) {
        assert.lengthOf(t.cookies, 2);
        assert.equal(t.cookies[0].key, 'broken_path');
        assert.equal(t.cookies[0].value, 'testme');
        assert.equal(t.cookies[1].key, "b");
        assert.equal(t.cookies[1].value, "2");
        assert.equal(t.cookies[1].path, "/");
      },
    },
  }
})
.addBatch({
  "Constructor":{
    topic: function () {
      return new Cookie({
        key: 'test',
        value: 'b',
        maxAge: 60
      });
    },
    'check for key property': function (c) {
      assert.ok(c);
      assert.equal(c.key, 'test');
    },
    'check for value property': function (c) {
      assert.equal(c.value, 'b');
    },
    'check for maxAge': function (c) {
      assert.equal(c.maxAge, 60);
    },
    'check for default values for unspecified properties': function (c) {
      assert.equal(c.expires, "Infinity");
      assert.equal(c.secure, false);
      assert.equal(c.httpOnly, false);
    }
  }
})
.addBatch({
  "allPaths option": {
    topic: function() {
      var cj = new CookieJar();
      var tasks = [];
      tasks.push(cj.setCookie.bind(cj, 'nopath_dom=qq; Path=/; Domain=example.com', 'http://example.com', {}));
      tasks.push(cj.setCookie.bind(cj, 'path_dom=qq; Path=/foo; Domain=example.com', 'http://example.com', {}));
      tasks.push(cj.setCookie.bind(cj, 'nopath_host=qq; Path=/', 'http://www.example.com', {}));
      tasks.push(cj.setCookie.bind(cj, 'path_host=qq; Path=/foo', 'http://www.example.com', {}));
      tasks.push(cj.setCookie.bind(cj, 'other=qq; Path=/', 'http://other.example.com/', {}));
      tasks.push(cj.setCookie.bind(cj, 'other2=qq; Path=/foo', 'http://other.example.com/foo', {}));
      var cb = this.callback;
      async.parallel(tasks, function(err,results) {
        cb(err, {cj:cj, cookies: results});
      });
    },
    "all set": function(t) {
      assert.equal(t.cookies.length, 6);
      assert.ok(t.cookies.every(function(c) { return !!c }));
    },
    "getting without allPaths": {
      topic: function(t) {
        var cb = this.callback;
        var cj = t.cj;
        cj.getCookies('http://www.example.com/', {}, function(err,cookies) {
          cb(err, {cj:cj, cookies:cookies});
        });
      },
      "found just two cookies": function(t) {
        assert.equal(t.cookies.length, 2);
      },
      "all are path=/": function(t) {
        assert.ok(t.cookies.every(function(c) { return c.path === '/' }));
      },
      "no 'other' cookies": function(t) {
        assert.ok(!t.cookies.some(function(c) { return (/^other/).test(c.name) }));
      },
    },
    "getting without allPaths for /foo": {
      topic: function(t) {
        var cb = this.callback;
        var cj = t.cj;
        cj.getCookies('http://www.example.com/foo', {}, function(err,cookies) {
          cb(err, {cj:cj, cookies:cookies});
        });
      },
      "found four cookies": function(t) {
        assert.equal(t.cookies.length, 4);
      },
      "no 'other' cookies": function(t) {
        assert.ok(!t.cookies.some(function(c) { return (/^other/).test(c.name) }));
      },
    },
    "getting with allPaths:true": {
      topic: function(t) {
        var cb = this.callback;
        var cj = t.cj;
        cj.getCookies('http://www.example.com/', {allPaths:true}, function(err,cookies) {
          cb(err, {cj:cj, cookies:cookies});
        });
      },
      "found four cookies": function(t) {
        assert.equal(t.cookies.length, 4);
      },
      "no 'other' cookies": function(t) {
        assert.ok(!t.cookies.some(function(c) { return (/^other/).test(c.name) }));
      },
    },
  }
})
.addBatch({
  "remove cookies": {
    topic: function() {
      var jar = new CookieJar();
      var cookie = Cookie.parse("a=b; Domain=example.com; Path=/");
      var cookie2 = Cookie.parse("a=b; Domain=foo.com; Path=/");
      var cookie3 = Cookie.parse("foo=bar; Domain=foo.com; Path=/");
      jar.setCookie(cookie, 'http://example.com/index.html', function(){});
      jar.setCookie(cookie2, 'http://foo.com/index.html', function(){});
      jar.setCookie(cookie3, 'http://foo.com/index.html', function(){});
      return jar;
    },
    "all from matching domain": function(jar){
      jar.store.removeCookies('example.com',null, function(err) {
        assert(err == null);

        jar.store.findCookies('example.com', null, function(err, cookies){
          assert(err == null);
          assert(cookies != null);
          assert(cookies.length === 0, 'cookie was not removed');
        });

        jar.store.findCookies('foo.com', null, function(err, cookies){
          assert(err == null);
          assert(cookies != null);
          assert(cookies.length === 2, 'cookies should not have been removed');
        });
      });
    },
    "from cookie store matching domain and key": function(jar){
      jar.store.removeCookie('foo.com', '/', 'foo', function(err) {
        assert(err == null);

        jar.store.findCookies('foo.com', null, function(err, cookies){
          assert(err == null);
          assert(cookies != null);
          assert(cookies.length === 1, 'cookie was not removed correctly');
          assert(cookies[0].key === 'a', 'wrong cookie was removed');
        });
      });
    }
  }
})
.addBatch({
  "Synchronous CookieJar": {
    "setCookieSync": {
      topic: function() {
        var jar = new CookieJar();
        var cookie = Cookie.parse("a=b; Domain=example.com; Path=/");
        cookie = jar.setCookieSync(cookie, 'http://example.com/index.html');
        return cookie;
      },
      "returns a copy of the cookie": function(cookie) {
        assert.instanceOf(cookie, Cookie);
      }
    },

    "setCookieSync strict parse error": {
      topic: function() {
        var jar = new CookieJar();
        var opts = { strict: true };
        try {
          jar.setCookieSync("farbe=weiß", 'http://example.com/index.html', opts);
          return false;
        } catch (e) {
          return e;
        }
      },
      "throws the error": function(err) {
        assert.instanceOf(err, Error);
        assert.equal(err.message, "Cookie failed to parse");
      }
    },

    "getCookiesSync": {
      topic: function() {
        var jar = new CookieJar();
        var url = 'http://example.com/index.html';
        jar.setCookieSync("a=b; Domain=example.com; Path=/", url);
        jar.setCookieSync("c=d; Domain=example.com; Path=/", url);
        return jar.getCookiesSync(url);
      },
      "returns the cookie array": function(err, cookies) {
        assert.ok(!err);
        assert.ok(Array.isArray(cookies));
        assert.lengthOf(cookies, 2);
        cookies.forEach(function(cookie) {
          assert.instanceOf(cookie, Cookie);
        });
      }
    },

    "getCookieStringSync": {
      topic: function() {
        var jar = new CookieJar();
        var url = 'http://example.com/index.html';
        jar.setCookieSync("a=b; Domain=example.com; Path=/", url);
        jar.setCookieSync("c=d; Domain=example.com; Path=/", url);
        return jar.getCookieStringSync(url);
      },
      "returns the cookie header string": function(err, str) {
        assert.ok(!err);
        assert.typeOf(str, 'string');
      }
    },

    "getSetCookieStringsSync": {
      topic: function() {
        var jar = new CookieJar();
        var url = 'http://example.com/index.html';
        jar.setCookieSync("a=b; Domain=example.com; Path=/", url);
        jar.setCookieSync("c=d; Domain=example.com; Path=/", url);
        return jar.getSetCookieStringsSync(url);
      },
      "returns the cookie header string": function(err, headers) {
        assert.ok(!err);
        assert.ok(Array.isArray(headers));
        assert.lengthOf(headers, 2);
        headers.forEach(function(header) {
          assert.typeOf(header, 'string');
        });
      }
    },
  }
})
.addBatch({
  "Synchronous API on async CookieJar": {
    topic: function() {
      return new tough.Store();
    },
    "setCookieSync": {
      topic: function(store) {
        var jar = new CookieJar(store);
        try {
          jar.setCookieSync("a=b", 'http://example.com/index.html');
          return false;
        } catch(e) {
          return e;
        }
      },
      "fails": function(err) {
        assert.instanceOf(err, Error);
        assert.equal(err.message,
                     'CookieJar store is not synchronous; use async API instead.');
      }
    },
    "getCookiesSync": {
      topic: function(store) {
        var jar = new CookieJar(store);
        try {
          jar.getCookiesSync('http://example.com/index.html');
          return false;
        } catch(e) {
          return e;
        }
      },
      "fails": function(err) {
        assert.instanceOf(err, Error);
        assert.equal(err.message,
                     'CookieJar store is not synchronous; use async API instead.');
      }
    },
    "getCookieStringSync": {
      topic: function(store) {
        var jar = new CookieJar(store);
        try {
          jar.getCookieStringSync('http://example.com/index.html');
          return false;
        } catch(e) {
          return e;
        }
      },
      "fails": function(err) {
        assert.instanceOf(err, Error);
        assert.equal(err.message,
                     'CookieJar store is not synchronous; use async API instead.');
      }
    },
    "getSetCookieStringsSync": {
      topic: function(store) {
        var jar = new CookieJar(store);
        try {
          jar.getSetCookieStringsSync('http://example.com/index.html');
          return false;
        } catch(e) {
          return e;
        }
      },
      "fails": function(err) {
        assert.instanceOf(err, Error);
        assert.equal(err.message,
                     'CookieJar store is not synchronous; use async API instead.');
      }
    },
  }
})
.export(module);
