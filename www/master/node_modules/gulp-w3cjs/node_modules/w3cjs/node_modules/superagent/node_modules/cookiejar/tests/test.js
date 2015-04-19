var Cookie=require("../cookiejar")
, CookieAccessInfo = Cookie.CookieAccessInfo
, CookieJar = Cookie.CookieJar
, Cookie = Cookie.Cookie

var test_jar = CookieJar();
test_jar.setCookies(
	"a=1;domain=.test.com;path=/"
	+":b=2;domain=test.com;path=/"
	+":c=3;domain=test.com;path=/;expires=January 1, 1970");
var cookies=test_jar.getCookies(CookieAccessInfo("test.com","/"))
console.log(
	cookies.length==2
	|| "Expires on setCookies fail"+cookies.length+"\n"+cookies.toString());
console.log(
    cookies.toValueString() == 'a=1;b=2'
    || "Cannot get value string of multiple cookies");
cookies=test_jar.getCookies(CookieAccessInfo("www.test.com","/"))
console.log(
	cookies.length==1
	|| "Wildcard domain fail"+cookies.length+"\n"+cookies.toString());
test_jar.setCookies("b=2;domain=test.com;path=/;expires=January 1, 1970");
cookies=test_jar.getCookies(CookieAccessInfo("test.com","/"))
console.log(
	cookies.length==1
	|| "Delete cookie fail"+cookies.length+"\n"+cookies.toString());
    
console.log(String(test_jar.getCookies(CookieAccessInfo("test.com","/"))))

cookie=Cookie("a=1;domain=test.com;path=/;HttpOnly");
console.log(cookie.noscript || "HttpOnly flag parsing failed\n"+cookie.toString());
