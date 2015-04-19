## Change Log

### v2.34.0 (2014/02/18 19:35 +00:00)
- [#781](https://github.com/mikeal/request/pull/781) simpler isReadStream function (@joaojeronimo)

- [#785](https://github.com/mikeal/request/pull/785) Provide ability to override content-type when `json` option used (@vvo)

- [#793](https://github.com/mikeal/request/pull/793) Adds content-length calculation when submitting forms using form-data li... (@Juul)

- [#802](https://github.com/mikeal/request/pull/802) Added the Apache license to the package.json. (@keskival)

- [#516](https://github.com/mikeal/request/pull/516) UNIX Socket URL Support (@lyuzashi)

- [#801](https://github.com/mikeal/request/pull/801) Ignore cookie parsing and domain errors (@lalitkapoor)


### v2.32.0 (2014/01/16 19:33 +00:00)
- [#757](https://github.com/mikeal/request/pull/757) require aws-sign2 (@mafintosh)

- [#744](https://github.com/mikeal/request/pull/744) Use Cookie.parse (@lalitkapoor)

- [#763](https://github.com/mikeal/request/pull/763) Upgrade tough-cookie to 0.10.0 (@stash)

- [#764](https://github.com/mikeal/request/pull/764) Case-insensitive authentication scheme (@bobyrizov)

- [#767](https://github.com/mikeal/request/pull/767) Use tough-cookie CookieJar sync API (@stash)


### v2.31.0 (2014/01/08 02:57 +00:00)
- [#736](https://github.com/mikeal/request/pull/736) Fix callback arguments documentation (@mmalecki)

- [#741](https://github.com/mikeal/request/pull/741) README example is using old cookie jar api (@emkay)

- [#742](https://github.com/mikeal/request/pull/742) Add note about JSON output body type (@iansltx)

- [#745](https://github.com/mikeal/request/pull/745) updating setCookie example to make it clear that the callback is required (@emkay)

- [#746](https://github.com/mikeal/request/pull/746) README: Markdown code highlight (@weakish)

- [#645](https://github.com/mikeal/request/pull/645) update twitter api url to v1.1 (@mick)


### v2.30.0 (2013/12/13 19:17 +00:00)
- [#732](https://github.com/mikeal/request/pull/732) JSHINT: Creating global 'for' variable. Should be 'for (var ...'. (@Fritz-Lium)

- [#730](https://github.com/mikeal/request/pull/730) better HTTP DIGEST support (@dai-shi)

- [#728](https://github.com/mikeal/request/pull/728) Fix TypeError when calling request.cookie (@scarletmeow)


### v2.29.0 (2013/12/06 20:05 +00:00)
- [#727](https://github.com/mikeal/request/pull/727) fix requester bug (@jchris)


### v2.28.0 (2013/12/04 19:42 +00:00)
- [#662](https://github.com/mikeal/request/pull/662) option.tunnel to explicitly disable tunneling (@seanmonstar)

- [#656](https://github.com/mikeal/request/pull/656) Test case for #304. (@diversario)

- [#666](https://github.com/mikeal/request/pull/666) make `ciphers` and `secureProtocol` to work in https request (@richarddong)

- [#683](https://github.com/mikeal/request/pull/683) Travis CI support (@Turbo87)

- [#690](https://github.com/mikeal/request/pull/690) Handle blank password in basic auth. (@diversario)

- [#694](https://github.com/mikeal/request/pull/694) Typo in README (@VRMink)

- [#696](https://github.com/mikeal/request/pull/696) Edited README.md for formatting and clarity of phrasing (@Zearin)

- [#630](https://github.com/mikeal/request/pull/630) Send random cnonce for HTTP Digest requests (@wprl)

- [#710](https://github.com/mikeal/request/pull/710) Fixing listing in callback part of docs. (@lukasz-zak)

- [#715](https://github.com/mikeal/request/pull/715) Request.multipart no longer crashes when header 'Content-type' present (@pastaclub)

- [#682](https://github.com/mikeal/request/pull/682) Optional dependencies (@Turbo87)

- [#719](https://github.com/mikeal/request/pull/719) Made a comment gender neutral. (@oztu)

- [#724](https://github.com/mikeal/request/pull/724) README.md: add custom HTTP Headers example. (@tcort)

- [#674](https://github.com/mikeal/request/pull/674) change cookie module,to tough-cookie.please check it . (@sxyizhiren)

- [#659](https://github.com/mikeal/request/pull/659) fix failure when running with NODE_DEBUG=request, and a test for that (@jrgm)


### v2.27.0 (2013/08/15 21:30 +00:00)
- [#619](https://github.com/mikeal/request/pull/619) decouple things a bit (@joaojeronimo)


### v2.26.0 (2013/08/07 16:31 +00:00)
- [#605](https://github.com/mikeal/request/pull/605) Only include ":" + pass in Basic Auth if it's defined (fixes #602) (@bendrucker)

- [#613](https://github.com/mikeal/request/pull/613) Fixes #583, moved initialization of self.uri.pathname  (@lexander)


### v2.24.0 (2013/07/23 20:51 +00:00)
- [#601](https://github.com/mikeal/request/pull/601) Fixed a small typo (@michalstanko)

- [#594](https://github.com/mikeal/request/pull/594) Emit complete event when there is no callback (@RomainLK)

- [#596](https://github.com/mikeal/request/pull/596) Global agent is being used when pool is specified (@Cauldrath)


### v2.23.0 (2013/07/23 02:44 +00:00)
- [#589](https://github.com/mikeal/request/pull/589) Prevent setting headers after they are sent (@wpreul)

- [#587](https://github.com/mikeal/request/pull/587) Global cookie jar disabled by default (@threepointone)


### v2.22.0 (2013/07/05 17:12 +00:00)
- [#542](https://github.com/mikeal/request/pull/542) Expose Request class (@regality)

- [#541](https://github.com/mikeal/request/pull/541) The exported request function doesn't have an auth method (@tschaub)

- [#564](https://github.com/mikeal/request/pull/564) Fix redirections (@criloz)

- [#568](https://github.com/mikeal/request/pull/568) use agentOptions to create agent when specified in request (@SamPlacette)

- [#581](https://github.com/mikeal/request/pull/581) Fix spelling of "ignoring." (@bigeasy)

- [#544](https://github.com/mikeal/request/pull/544) Update http-signature version. (@davidlehn)


### v2.21.0 (2013/04/30 21:28 +00:00)
- [#529](https://github.com/mikeal/request/pull/529) dependencies versions bump (@jodaka)

- [#521](https://github.com/mikeal/request/pull/521) Improving test-localAddress.js (@noway421)

- [#503](https://github.com/mikeal/request/pull/503) Fix basic auth for passwords that contain colons (@tonistiigi)

- [#497](https://github.com/mikeal/request/pull/497) Added redirect event (@Cauldrath)

- [#532](https://github.com/mikeal/request/pull/532) fix typo (@fredericosilva)

- [#536](https://github.com/mikeal/request/pull/536) Allow explicitly empty user field for basic authentication. (@mikeando)


### v2.17.0 (2013/04/22 15:52 +00:00)
- [#19](https://github.com/mikeal/request/pull/19) Request is unusable without native ssl support in node (@davglass)

- [#31](https://github.com/mikeal/request/pull/31) Error on piping a request to a destination (@tobowers)

- [#35](https://github.com/mikeal/request/pull/35) The "end" event isn't emitted for some responses (@voxpelli)

- [#45](https://github.com/mikeal/request/pull/45) Added timeout option (@mbrevoort)

- [#66](https://github.com/mikeal/request/pull/66) Do not overwrite established content-type headers for read stream deliver (@voodootikigod)

- [#67](https://github.com/mikeal/request/pull/67) fixed global variable leaks (@aheckmann)

- [#69](https://github.com/mikeal/request/pull/69) Flatten chunked requests properly (@isaacs)

- [#73](https://github.com/mikeal/request/pull/73) Fix #71 Respect the strictSSL flag (@isaacs)

- [#70](https://github.com/mikeal/request/pull/70) add test script to package.json (@isaacs)

- [#76](https://github.com/mikeal/request/pull/76) Bug when a request fails and a timeout is set (@Marsup)

- [#78](https://github.com/mikeal/request/pull/78) Don't try to do strictSSL for non-ssl connections (@isaacs)

- [#79](https://github.com/mikeal/request/pull/79) Proxy auth bug (@isaacs)

- [#81](https://github.com/mikeal/request/pull/81) Enhance redirect handling (@danmactough)

- [#96](https://github.com/mikeal/request/pull/96) Authless parsed url host support (@isaacs)

- [#84](https://github.com/mikeal/request/pull/84) Document strictSSL option (@isaacs)

- [#97](https://github.com/mikeal/request/pull/97) Typo in previous pull causes TypeError in non-0.5.11 versions (@isaacs)

- [#53](https://github.com/mikeal/request/pull/53) Parse json: Issue #51 (@benatkin)

- [#102](https://github.com/mikeal/request/pull/102) Implemented cookies - closes issue 82: https://github.com/mikeal/request/issues/82 (@alessioalex)

- [#105](https://github.com/mikeal/request/pull/105) added test for proxy option. (@dominictarr)

- [#86](https://github.com/mikeal/request/pull/86) Can't post binary to multipart requests (@kkaefer)

- [#110](https://github.com/mikeal/request/pull/110) Update to Iris Couch URL (@jhs)

- [#117](https://github.com/mikeal/request/pull/117) Remove the global `i` (@3rd-Eden)

- [#121](https://github.com/mikeal/request/pull/121) Another patch for cookie handling regression (@jhurliman)

- [#104](https://github.com/mikeal/request/pull/104) Cookie handling contains bugs (@janjongboom)

- [#112](https://github.com/mikeal/request/pull/112) Support using a custom http-like module (@jhs)

- [#132](https://github.com/mikeal/request/pull/132) return the body as a Buffer when encoding is set to null (@jahewson)

- [#135](https://github.com/mikeal/request/pull/135) host vs hostname (@iangreenleaf)

- [#133](https://github.com/mikeal/request/pull/133) Fixed cookies parsing (@afanasy)

- [#144](https://github.com/mikeal/request/pull/144) added "form" option to readme (@petejkim)

- [#146](https://github.com/mikeal/request/pull/146) Multipart should respect content-type if previously set (@apeace)

- [#148](https://github.com/mikeal/request/pull/148) Retry Agent (@thejh)

- [#90](https://github.com/mikeal/request/pull/90) add option followAllRedirects to follow post/put redirects (@jroes)

- [#162](https://github.com/mikeal/request/pull/162) Fix issue #159 (@dpetukhov)

- [#161](https://github.com/mikeal/request/pull/161) Fix cookie jar/headers.cookie collision (#125) (@papandreou)

- [#168](https://github.com/mikeal/request/pull/168) Picking off an EasyFix by adding some missing mimetypes. (@serby)

- [#170](https://github.com/mikeal/request/pull/170) can't create a cookie in a wrapped request (defaults) (@fabianonunes)

- [#179](https://github.com/mikeal/request/pull/179) fix to add opts in .pipe(stream, opts) (@substack)

- [#180](https://github.com/mikeal/request/pull/180) Modified the post, put, head and del shortcuts to support uri optional param (@twilson63)

- [#177](https://github.com/mikeal/request/pull/177) Issue #173 Support uri as first and optional config as second argument (@twilson63)

- [#182](https://github.com/mikeal/request/pull/182) Fix request.defaults to support (uri, options, callback) api (@twilson63)

- [#176](https://github.com/mikeal/request/pull/176) Querystring option (@csainty)

- [#188](https://github.com/mikeal/request/pull/188) Add abort support to the returned request (@itay)

- [#193](https://github.com/mikeal/request/pull/193) Fixes GH-119 (@goatslacker)

- [#197](https://github.com/mikeal/request/pull/197) Make ForeverAgent work with HTTPS (@isaacs)

- [#198](https://github.com/mikeal/request/pull/198) Bugfix on forever usage of util.inherits (@isaacs)

- [#199](https://github.com/mikeal/request/pull/199) Tunnel (@isaacs)

- [#203](https://github.com/mikeal/request/pull/203) Fix cookie and redirect bugs and add auth support for HTTPS tunnel (@milewise)

- [#217](https://github.com/mikeal/request/pull/217) need to use Authorization (titlecase) header with Tumblr OAuth (@visnup)

- [#224](https://github.com/mikeal/request/pull/224) Multipart content-type change (@janjongboom)

- [#211](https://github.com/mikeal/request/pull/211) Replace all occurrences of special chars in RFC3986 (@chriso)

- [#240](https://github.com/mikeal/request/pull/240) don't error when null is passed for options (@polotek)

- [#243](https://github.com/mikeal/request/pull/243) Dynamic boundary (@zephrax)

- [#246](https://github.com/mikeal/request/pull/246) Fixing the set-cookie header (@jeromegn)

- [#260](https://github.com/mikeal/request/pull/260) fixed just another leak of 'i' (@sreuter)

- [#255](https://github.com/mikeal/request/pull/255) multipart allow body === '' ( the empty string ) (@Filirom1)

- [#261](https://github.com/mikeal/request/pull/261) Setting 'pool' to 'false' does NOT disable Agent pooling (@timshadel)

- [#262](https://github.com/mikeal/request/pull/262) JSON test should check for equality (@timshadel)

- [#265](https://github.com/mikeal/request/pull/265) uncaughtException when redirected to invalid URI (@naholyr)

- [#263](https://github.com/mikeal/request/pull/263) Bug in OAuth key generation for sha1 (@nanodocumet)

- [#268](https://github.com/mikeal/request/pull/268) I'm not OCD seriously (@TehShrike)

- [#273](https://github.com/mikeal/request/pull/273) Pipe back pressure issue (@mafintosh)

- [#279](https://github.com/mikeal/request/pull/279) fix tests with boundary by injecting boundry from header (@benatkin)

- [#241](https://github.com/mikeal/request/pull/241) Composability updates suggested by issue #239 (@polotek)

- [#284](https://github.com/mikeal/request/pull/284) Remove stray `console.log()` call in multipart generator. (@bcherry)

- [#272](https://github.com/mikeal/request/pull/272) Boundary begins with CRLF? (@proksoup)

- [#207](https://github.com/mikeal/request/pull/207) Fix #206 Change HTTP/HTTPS agent when redirecting between protocols (@isaacs)

- [#280](https://github.com/mikeal/request/pull/280) Like in node.js print options if NODE_DEBUG contains the word request (@Filirom1)

- [#290](https://github.com/mikeal/request/pull/290) A test for #289 (@isaacs)

- [#293](https://github.com/mikeal/request/pull/293) Allow parser errors to bubble up to request (@mscdex)

- [#317](https://github.com/mikeal/request/pull/317) Workaround for #313 (@isaacs)

- [#318](https://github.com/mikeal/request/pull/318) Pass servername to tunneling secure socket creation (@isaacs)

- [#326](https://github.com/mikeal/request/pull/326) Do not try to remove listener from an undefined connection (@strk)

- [#320](https://github.com/mikeal/request/pull/320) request.defaults() doesn't need to wrap jar() (@StuartHarris)

- [#343](https://github.com/mikeal/request/pull/343) Allow AWS to work in more situations, added a note in the README on its usage (@nlf)

- [#332](https://github.com/mikeal/request/pull/332) Fix #296 - Only set Content-Type if body exists (@Marsup)

- [#355](https://github.com/mikeal/request/pull/355) stop sending erroneous headers on redirected requests (@azylman)

- [#360](https://github.com/mikeal/request/pull/360) Delete self._form along with everything else on redirect (@jgautier)

- [#361](https://github.com/mikeal/request/pull/361) Don't create a Content-Length header if we already have it set (@danjenkins)

- [#362](https://github.com/mikeal/request/pull/362) Running `rfc3986` on `base_uri` in `oauth.hmacsign` instead of just `encodeURIComponent` (@jeffmarshall)

- [#363](https://github.com/mikeal/request/pull/363) rfc3986 on base_uri, now passes tests (@jeffmarshall)

- [#344](https://github.com/mikeal/request/pull/344) Make AWS auth signing find headers correctly (@nlf)

- [#369](https://github.com/mikeal/request/pull/369) Don't remove x_auth_mode for Twitter reverse auth (@drudge)

- [#370](https://github.com/mikeal/request/pull/370) Twitter reverse auth uses x_auth_mode not x_auth_type (@drudge)

- [#374](https://github.com/mikeal/request/pull/374) Correct Host header for proxy tunnel CONNECT (@ypocat)

- [#375](https://github.com/mikeal/request/pull/375) Fix for missing oauth_timestamp parameter (@jplock)

- [#376](https://github.com/mikeal/request/pull/376) Headers lost on redirect (@kapetan)

- [#380](https://github.com/mikeal/request/pull/380) Fixes missing host header on retried request when using forever agent (@mac-)

- [#381](https://github.com/mikeal/request/pull/381) Resolving "Invalid signature. Expected signature base string: " (@landeiro)

- [#398](https://github.com/mikeal/request/pull/398) Add more reporting to tests (@mmalecki)

- [#403](https://github.com/mikeal/request/pull/403) Optimize environment lookup to happen once only (@mmalecki)

- [#415](https://github.com/mikeal/request/pull/415) Fixed a typo. (@jerem)

- [#430](https://github.com/mikeal/request/pull/430) Respect specified {Host,host} headers, not just {host} (@andrewschaaf)

- [#338](https://github.com/mikeal/request/pull/338) Add more auth options, including digest support (@nylen)

- [#448](https://github.com/mikeal/request/pull/448) Convenience method for PATCH (@mloar)

- [#413](https://github.com/mikeal/request/pull/413) rename googledoodle.png to .jpg (@nfriedly)

- [#454](https://github.com/mikeal/request/pull/454) Destroy the response if present when destroying the request (clean merge) (@mafintosh)

- [#429](https://github.com/mikeal/request/pull/429) Copy options before adding callback. (@nrn)

- [#462](https://github.com/mikeal/request/pull/462) if query params are empty, then request path shouldn't end with a '?' (merges cleanly now) (@jaipandya)

- [#471](https://github.com/mikeal/request/pull/471) Using querystring library from visionmedia (@kbackowski)

- [#473](https://github.com/mikeal/request/pull/473) V0.10 compat (@isaacs)

- [#475](https://github.com/mikeal/request/pull/475) Use `unescape` from `querystring` (@shimaore)

- [#479](https://github.com/mikeal/request/pull/479) Changing so if Accept header is explicitly set, sending json does not ov... (@RoryH)

- [#490](https://github.com/mikeal/request/pull/490) Empty response body (3-rd argument) must be passed to callback as an empty string (@Olegas)

- [#498](https://github.com/mikeal/request/pull/498) Moving response emit above setHeaders on destination streams (@kenperkins)

- [#512](https://github.com/mikeal/request/pull/512) Make password optional to support the format: http://username@hostname/ (@pajato1)

- [#508](https://github.com/mikeal/request/pull/508) Honor the .strictSSL option when using proxies (tunnel-agent) (@jhs)

- [#519](https://github.com/mikeal/request/pull/519) Update internal path state on post-creation QS changes (@jblebrun)

- [#520](https://github.com/mikeal/request/pull/520) Fixing test-tunnel.js (@noway421)

- [#523](https://github.com/mikeal/request/pull/523) Updating dependencies (@noway421)

- [#510](https://github.com/mikeal/request/pull/510) Add HTTP Signature support. (@davidlehn)

- [#456](https://github.com/mikeal/request/pull/456) hawk 0.9.0 (@hueniverse)

- [#460](https://github.com/mikeal/request/pull/460) hawk 0.10.0 (@hueniverse)

- [#444](https://github.com/mikeal/request/pull/444) protect against double callbacks on error path (@spollack)

- [#322](https://github.com/mikeal/request/pull/322) Fix + test for piped into request bumped into redirect. #321 (@alexindigo)

- [#513](https://github.com/mikeal/request/pull/513) add 'localAddress' support (@yyfrankyy)

- [#249](https://github.com/mikeal/request/pull/249) Fix for the fix of your (closed) issue #89 where self.headers[content-length] is set to 0 for all methods (@sethbridges)

- [#502](https://github.com/mikeal/request/pull/502) Fix POST (and probably other) requests that are retried after 401 Unauthorized (@nylen)

- [#282](https://github.com/mikeal/request/pull/282) OAuth Authorization header contains non-"oauth_" parameters (@jplock)

- [#388](https://github.com/mikeal/request/pull/388) Ensure "safe" toJSON doesn't break EventEmitters (@othiym23)

- [#214](https://github.com/mikeal/request/pull/214) documenting additional behavior of json option (@jphaas)

- [#310](https://github.com/mikeal/request/pull/310) Twitter Oauth Stuff Out of Date; Now Updated (@joemccann)

- [#433](https://github.com/mikeal/request/pull/433) Added support for HTTPS cert & key (@indexzero)

- [#461](https://github.com/mikeal/request/pull/461) Strip the UTF8 BOM from a UTF encoded response (@kppullin)


### v1.2.0 (2011/01/30 22:04 +00:00)
- [#3](https://github.com/mikeal/request/pull/3) JSON body (@Stanley)