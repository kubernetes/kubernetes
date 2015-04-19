# useragent - high performance user agent parser for Node.js

Useragent originated as port of [browserscope.org][browserscope]'s user agent
parser project also known as ua-parser. Useragent allows you to parse user agent
string with high accuracy by using hand tuned dedicated regular expressions for
browser matching. This database is needed to ensure that every browser is
correctly parsed as every browser vendor implements it's own user agent schema.
This is why regular user agent parsers have major issues because they will
most likely parse out the wrong browser name or confuse the render engine version
with the actual version of the browser.

---

### Build status [![BuildStatus](https://secure.travis-ci.org/3rd-Eden/useragent.png?branch=master)](http://travis-ci.org/3rd-Eden/useragent)

---

### High performance

The module has been developed with a benchmark driven approach. It has a
pre-compiled library that contains all the Regular Expressions and uses deferred
or on demand parsing for Operating System and device information. All this
engineering effort has been worth it as [this benchmark shows][benchmark]:

```
Starting the benchmark, parsing 62 useragent strings per run

Executed benchmark against node module: "useragent"
Count (61), Cycles (5), Elapsed (5.559), Hz (1141.3739447904327)

Executed benchmark against node module: "useragent_parser"
Count (29), Cycles (3), Elapsed (5.448), Hz (545.6817291171243)

Executed benchmark against node module: "useragent-parser"
Count (16), Cycles (4), Elapsed (5.48), Hz (304.5373431830105)

Executed benchmark against node module: "ua-parser"
Count (54), Cycles (3), Elapsed (5.512), Hz (1018.7561434659247)

Module: "useragent" is the user agent fastest parser.
```

---

### Installation

Installation is done using the Node Package Manager (NPM). If you don't have
NPM installed on your system you can download it from
[npmjs.org][npm]

```
npm install useragent --save
```

The `--save` flag tells NPM to automatically add it to your `package.json` file.

---

### API


Include the `useragent` parser in you node.js application:

```js
var useragent = require('useragent');
```

The `useragent` library allows you do use the automatically installed RegExp
library or you can fetch it live from the remote servers. So if you are
paranoid and always want your RegExp library to be up to date to match with
agent the widest range of `useragent` strings you can do:

```js
var useragent = require('useragent');
useragent(true);
```

This will async load the database from the server and compile it to a proper
JavaScript supported format. If it fails to compile or load it from the remote
location it will just fall back silently to the shipped version. If you want to
use this feature you need to add `yamlparser` and `request` to your package.json

```
npm install yamlparser --save
npm install request --save
```

#### useragent.parse(useragent string[, js useragent]);

This is the actual user agent parser, this is where all the magic is happening.
The function accepts 2 arguments, both should be a `string`. The first argument
should the user agent string that is known on the server from the
`req.headers.useragent` header. The other argument is optional and should be
the user agent string that you see in the browser, this can be send from the
browser using a xhr request or something like this. This allows you detect if
the user is browsing the web using the `Chrome Frame` extension.

The parser returns a Agent instance, this allows you to output user agent
information in different predefined formats. See the Agent section for more
information.

```js
var agent = useragent.parse(req.headers['user-agent']);

// example for parsing both the useragent header and a optional js useragent
var agent2 = useragent.parse(req.headers['user-agent'], req.query.jsuseragent);
```

The parse method returns a `Agent` instance which contains all details about the
user agent. See the Agent section of the API documentation for the available
methods.

#### useragent.lookup(useragent string[, js useragent]);

This provides the same functionality as above, but it caches the user agent
string and it's parsed result in memory to provide faster lookups in the
future. This can be handy if you expect to parse a lot of user agent strings.

It uses the same arguments as the `useragent.parse` method and returns exactly
the same result, but it's just cached.

```js
var agent = useragent.lookup(req.headers['user-agent']);
```

And this is a serious performance improvement as shown in this benchmark:

```
Executed benchmark against method: "useragent.parse"
Count (49), Cycles (3), Elapsed (5.534), Hz (947.6844321931629)

Executed benchmark against method: "useragent.lookup"
Count (11758), Cycles (3), Elapsed (5.395), Hz (229352.03831239208)
```

#### useragent.fromJSON(obj);

Transforms the JSON representation of a `Agent` instance back in to a working
`Agent` instance

```js
var agent = useragent.parse(req.headers['user-agent'])
  , another = useragent.fromJSON(JSON.stringify(agent));

console.log(agent == another);
```

#### useragent.is(useragent string).browsername;

This api provides you with a quick and dirty browser lookup. The underlying
code is usually found on client side scripts so it's not the same quality as
our `useragent.parse` method but it might be needed for legacy reasons.

`useragent.is` returns a object with potential matched browser names

```js
useragent.is(req.headers['user-agent']).firefox // true
useragent.is(req.headers['user-agent']).safari // false
var ua = useragent.is(req.headers['user-agent'])

// the object
{
  version: '3'
  webkit: false
  opera: false
  ie: false
  chrome: false
  safari: false
  mobile_safari: false
  firefox: true
  mozilla: true
  android: false
}
```

---

### Agents, OperatingSystem and Device instances

Most of the methods mentioned above return a Agent instance. The Agent exposes
the parsed out information from the user agent strings. This allows us to
extend the agent with more methods that do not necessarily need to be in the
core agent instance, allowing us to expose a plugin interface for third party
developers and at the same time create a uniform interface for all versioning.

The Agent has the following property

- `family` The browser family, or browser name, it defaults to Other.
- `major` The major version number of the family, it defaults to 0.
- `minor` The minor version number of the family, it defaults to 0.
- `patch` The patch version number of the family, it defaults to 0.

In addition to the properties mentioned above, it also has 2 special properties,
which are:

- `os` OperatingSystem instance
- `device` Device instance

When you access those 2 properties the agent will do on demand parsing of the
Operating System or/and Device information.

The OperatingSystem has the same properties as the Agent, for the Device we
don't have any versioning information available, so only the `family` property is
set there. If we cannot find the family, they will default to `Other`.

The following methods are available:

#### Agent.toAgent();

Returns the family and version number concatinated in a nice human readable
string.

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.toAgent(); // 'Chrome 15.0.874'
```

#### Agent.toString();

Returns the results of the `Agent.toAgent()` but also adds the parsed operating
system to the string in a human readable format.

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.toString(); // 'Chrome 15.0.874 / Mac OS X 10.8.1'

// as it's a to string method you can also concat it with another string
'your useragent is ' + agent;
// 'your useragent is Chrome 15.0.874 / Mac OS X 10.8.1'
```
#### Agent.toVersion();

Returns the version of the browser in a human readable string.

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.toVersion(); // '15.0.874'
```

#### Agent.toJSON();

Generates a JSON representation of the Agent. By using the `toJSON` method we
automatically allow it to be stringified when supplying as to the
`JSON.stringify` method.

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.toJSON(); // returns an object

JSON.stringify(agent);
```

#### OperatingSystem.toString();

Generates a stringified version of operating system;

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.os.toString(); // 'Mac OSX 10.8.1'
```

#### OperatingSystem.toVersion();

Generates a stringified version of operating system's version;

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.os.toVersion(); // '10.8.1'
```

#### OperatingSystem.toJSON();

Generates a JSON representation of the OperatingSystem. By using the `toJSON`
method we automatically allow it to be stringified when supplying as to the
`JSON.stringify` method.

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.os.toJSON(); // returns an object

JSON.stringify(agent.os);
```

#### Device.toString();

Generates a stringified version of device;

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.device.toString(); // 'Asus A100'
```

#### Device.toVersion();

Generates a stringified version of device's version;

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.device.toVersion(); // '' , no version found but could also be '0.0.0'
```

#### Device.toJSON();

Generates a JSON representation of the Device. By using the `toJSON` method we
automatically allow it to be stringified when supplying as to the
`JSON.stringify` method.

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.device.toJSON(); // returns an object

JSON.stringify(agent.device);
```

### Adding more features to the useragent

As I wanted to keep the core of the user agent parser as clean and fast as
possible I decided to move some of the initially planned features to a new
`plugin` file.

These extensions to the Agent prototype can be loaded by requiring the
`useragent/features` file:

```js
var useragent = require('useragent');
require('useragent/features');
```

The initial release introduces 1 new method, satisfies, which allows you to see
if the version number of the browser satisfies a certain range. It uses the
semver library to do all the range calculations but here is a small summary of
the supported range styles:

* `>1.2.3` Greater than a specific version.
* `<1.2.3` Less than.
* `1.2.3 - 2.3.4` := `>=1.2.3 <=2.3.4`.
* `~1.2.3` := `>=1.2.3 <1.3.0`.
* `~1.2` := `>=1.2.0 <2.0.0`.
* `~1` := `>=1.0.0 <2.0.0`.
* `1.2.x` := `>=1.2.0 <1.3.0`.
* `1.x` := `>=1.0.0 <2.0.0`.

As it requires the `semver` module to function you need to install it
seperately:

```
npm install semver --save
```

#### Agent.satisfies('range style here');

Check if the agent matches the supplied range.

```js
var agent = useragent.parse(req.headers['user-agent']);
agent.satisfies('15.x || >=19.5.0 || 25.0.0 - 17.2.3'); // true
agent.satisfies('>16.12.0'); // false
```
---

### Migrations

For small changes between version please review the [changelog][changelog].

#### Upgrading from 1.10 to 2.0.0

- `useragent.fromAgent` has been removed.
- `agent.toJSON` now returns an Object, use `JSON.stringify(agent)` for the old
  behaviour.
- `agent.os` is now an `OperatingSystem` instance with version numbers. If you
  still a string only representation do `agent.os.toString()`.
- `semver` has been removed from the dependencies, so if you are using the
  `require('useragent/features')` you need to add it to your own dependencies

#### Upgrading from 0.1.2 to 1.0.0

- `useragent.browser(ua)` has been renamed to `useragent.is(ua)`.
- `useragent.parser(ua, jsua)` has been renamed to `useragent.parse(ua, jsua)`.
- `result.pretty()` has been renamed to `result.toAgent()`.
- `result.V1` has been renamed to `result.major`.
- `result.V2` has been renamed to `result.minor`.
- `result.V3` has been renamed to `result.patch`.
- `result.prettyOS()` has been removed.
- `result.match` has been removed.

---

### License

MIT

[browserscope]: http://www.browserscope.org/
[benchmark]: /3rd-Eden/useragent/blob/master/benchmark/run.js
[changelog]: /3rd-Eden/useragent/blob/master/CHANGELOG.md
[npm]: http://npmjs.org
