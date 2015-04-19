#CookieJar

Simple robust cookie library

##Exports


###CookieAccessInfo(domain,path,secure,script)
    class to determine matching qualities of a cookie

#####Properties
* String domain - domain to match
* String path - path to match
* Boolean secure - access is secure (ssl generally)
* Boolean script - access is from a script


###Cookie(cookiestr_or_cookie)
    turns input into a Cookie (singleton if given a Cookie)

#####Properties
* String name - name of the cookie
* String value - string associated with the cookie
* String domain - domain to match (on a cookie a '.' at the start means a wildcard matching anything ending in the rest)
* String path - base path to match (matches any path starting with this '/' is root)
* Boolean noscript - if it should be kept from scripts
* Boolean secure - should it only be transmitted over secure means
* Number expiration_date - number of millis since 1970 at which this should be removed

#####Methods
* String toString() - the __set-cookie:__ string for this cookie
* String toValueString() - the __cookie:__ string for this cookie
* Cookie parse(cookiestr) - parses the string onto this cookie or a new one if called directly
* Boolean matches(access_info) - returns true if the access_info allows retrieval of this cookie
* Boolean collidesWith(cookie) - returns true if the cookies cannot exist in the same space (domain and path match)


###CookieJar()
    class to hold numerous cookies from multiple domains correctly

#####Methods
* Cookie setCookie(cookie) - add a cookie to the jar
* Cookie[] setCookies(cookiestr_or_list) - add a large number of cookies to the jar
* Cookie getCookie(cookie_name,access_info) - get a cookie with the name and access_info matching
* Cookie[] getCookies(access_info) - grab all cookies matching this access_info