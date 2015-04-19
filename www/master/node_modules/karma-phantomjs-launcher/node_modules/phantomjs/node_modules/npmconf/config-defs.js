// defaults, types, and shorthands.


var path = require("path")
  , url = require("url")
  , Stream = require("stream").Stream
  , semver = require("semver")
  , stableFamily = semver.parse(process.version)
  , nopt = require("nopt")
  , os = require("os")
  , osenv = require("osenv")

var log
try {
  log = require("npmlog")
} catch (er) {
  var util = require("util")
  log = { warn: function (m) {
    console.warn(m + ' ' + util.format.apply(util, [].slice.call(arguments, 1)))
  } }
}

exports.Octal = Octal
function Octal () {}
function validateOctal (data, k, val) {
  // must be either an integer or an octal string.
  if (typeof val === "number") {
    data[k] = val
    return true
  }

  if (typeof val === "string") {
    if (val.charAt(0) !== "0" || isNaN(val)) return false
    data[k] = parseInt(val, 8).toString(8)
  }
}

function validateSemver (data, k, val) {
  if (!semver.valid(val)) return false
  data[k] = semver.valid(val)
}

function validateTag (data, k, val) {
  val = ('' + val).trim()
  if (!val || semver.validRange(val)) return false
  data[k] = val
}

function validateStream (data, k, val) {
  if (!(val instanceof Stream)) return false
  data[k] = val
}

nopt.typeDefs.semver = { type: semver, validate: validateSemver }
nopt.typeDefs.Octal = { type: Octal, validate: validateOctal }
nopt.typeDefs.Stream = { type: Stream, validate: validateStream }

// Don't let --tag=1.2.3 ever be a thing
var tag = {}
nopt.typeDefs.tag = { type: tag, validate: validateTag }

nopt.invalidHandler = function (k, val, type) {
  log.warn("invalid config", k + "=" + JSON.stringify(val))

  if (Array.isArray(type)) {
    if (type.indexOf(url) !== -1) type = url
    else if (type.indexOf(path) !== -1) type = path
  }

  switch (type) {
    case tag:
      log.warn("invalid config", "Tag must not be a SemVer range")
      break
    case Octal:
      log.warn("invalid config", "Must be octal number, starting with 0")
      break
    case url:
      log.warn("invalid config", "Must be a full url with 'http://'")
      break
    case path:
      log.warn("invalid config", "Must be a valid filesystem path")
      break
    case Number:
      log.warn("invalid config", "Must be a numeric value")
      break
    case Stream:
      log.warn("invalid config", "Must be an instance of the Stream class")
      break
  }
}

if (!stableFamily || (+stableFamily.minor % 2)) stableFamily = null
else stableFamily = stableFamily.major + "." + stableFamily.minor

var defaults

var temp = osenv.tmpdir()
var home = osenv.home()

var uidOrPid = process.getuid ? process.getuid() : process.pid

if (home) process.env.HOME = home
else home = path.resolve(temp, "npm-" + uidOrPid)

var cacheExtra = process.platform === "win32" ? "npm-cache" : ".npm"
var cacheRoot = process.platform === "win32" && process.env.APPDATA || home
var cache = path.resolve(cacheRoot, cacheExtra)


var globalPrefix
Object.defineProperty(exports, "defaults", {get: function () {
  if (defaults) return defaults

  if (process.env.PREFIX) {
    globalPrefix = process.env.PREFIX
  } else if (process.platform === "win32") {
    // c:\node\node.exe --> prefix=c:\node\
    globalPrefix = path.dirname(process.execPath)
  } else {
    // /usr/local/bin/node --> prefix=/usr/local
    globalPrefix = path.dirname(path.dirname(process.execPath))

    // destdir only is respected on Unix
    if (process.env.DESTDIR) {
      globalPrefix = path.join(process.env.DESTDIR, globalPrefix)
    }
  }

  return defaults =
    { "always-auth" : false
    , "bin-links" : true
    , browser : null

    , ca: null
    , cafile: null

    , cache : cache

    , "cache-lock-stale": 60000
    , "cache-lock-retries": 10
    , "cache-lock-wait": 10000

    , "cache-max": Infinity
    , "cache-min": 10

    , cert: null

    , color : true
    , depth: Infinity
    , description : true
    , dev : false
    , editor : osenv.editor()
    , "engine-strict": false
    , force : false

    , "fetch-retries": 2
    , "fetch-retry-factor": 10
    , "fetch-retry-mintimeout": 10000
    , "fetch-retry-maxtimeout": 60000

    , git: "git"
    , "git-tag-version": true

    , global : false
    , globalconfig : path.resolve(globalPrefix, "etc", "npmrc")
    , group : process.platform === "win32" ? 0
            : process.env.SUDO_GID || (process.getgid && process.getgid())
    , heading: "npm"
    , "ignore-scripts": false
    , "init-module": path.resolve(home, ".npm-init.js")
    , "init.author.name" : ""
    , "init.author.email" : ""
    , "init.author.url" : ""
    , "init.version": "1.0.0"
    , "init.license": "ISC"
    , json: false
    , key: null
    , link: false
    , "local-address" : undefined
    , loglevel : "warn"
    , logstream : process.stderr
    , long : false
    , message : "%s"
    , "node-version" : process.version
    , npat : false
    , "onload-script" : false
    , optional: true
    , parseable : false
    , prefix : globalPrefix
    , production: process.env.NODE_ENV === "production"
    , "proprietary-attribs": true
    , proxy : process.env.HTTP_PROXY || process.env.http_proxy || null
    , "https-proxy" : process.env.HTTPS_PROXY || process.env.https_proxy ||
                      process.env.HTTP_PROXY || process.env.http_proxy || null
    , "user-agent" : "npm/{npm-version} "
                     + "node/{node-version} "
                     + "{platform} "
                     + "{arch}"
    , "rebuild-bundle" : true
    , registry : "https://registry.npmjs.org/"
    , rollback : true
    , save : false
    , "save-bundle": false
    , "save-dev" : false
    , "save-exact" : false
    , "save-optional" : false
    , "save-prefix": "^"
    , scope : ""
    , searchopts: ""
    , searchexclude: null
    , searchsort: "name"
    , shell : osenv.shell()
    , shrinkwrap: true
    , "sign-git-tag": false
    , spin: true
    , "strict-ssl": true
    , tag : "latest"
    , tmp : temp
    , unicode : true
    , "unsafe-perm" : process.platform === "win32"
                    || process.platform === "cygwin"
                    || !( process.getuid && process.setuid
                       && process.getgid && process.setgid )
                    || process.getuid() !== 0
    , usage : false
    , user : process.platform === "win32" ? 0 : "nobody"
    , userconfig : path.resolve(home, ".npmrc")
    , umask: process.umask ? process.umask() : parseInt("022", 8)
    , version : false
    , versions : false
    , viewer: process.platform === "win32" ? "browser" : "man"

    , _exit : true
    }
}})

exports.types =
  { "always-auth" : Boolean
  , "bin-links": Boolean
  , browser : [null, String]
  , ca: [null, String, Array]
  , cafile : path
  , cache : path
  , "cache-lock-stale": Number
  , "cache-lock-retries": Number
  , "cache-lock-wait": Number
  , "cache-max": Number
  , "cache-min": Number
  , cert: [null, String]
  , color : ["always", Boolean]
  , depth : Number
  , description : Boolean
  , dev : Boolean
  , editor : String
  , "engine-strict": Boolean
  , force : Boolean
  , "fetch-retries": Number
  , "fetch-retry-factor": Number
  , "fetch-retry-mintimeout": Number
  , "fetch-retry-maxtimeout": Number
  , git: String
  , "git-tag-version": Boolean
  , global : Boolean
  , globalconfig : path
  , group : [Number, String]
  , "https-proxy" : [null, url]
  , "user-agent" : String
  , "heading": String
  , "ignore-scripts": Boolean
  , "init-module": path
  , "init.author.name" : String
  , "init.author.email" : String
  , "init.author.url" : ["", url]
  , "init.license": String
  , "init.version": semver
  , json: Boolean
  , key: [null, String]
  , link: Boolean
  // local-address must be listed as an IP for a local network interface
  // must be IPv4 due to node bug
  , "local-address" : getLocalAddresses()
  , loglevel : ["silent","error","warn","http","info","verbose","silly"]
  , logstream : Stream
  , long : Boolean
  , message: String
  , "node-version" : [null, semver]
  , npat : Boolean
  , "onload-script" : [null, String]
  , optional: Boolean
  , parseable : Boolean
  , prefix: path
  , production: Boolean
  , "proprietary-attribs": Boolean
  , proxy : [null, url]
  , "rebuild-bundle" : Boolean
  , registry : [null, url]
  , rollback : Boolean
  , save : Boolean
  , "save-bundle": Boolean
  , "save-dev" : Boolean
  , "save-exact" : Boolean
  , "save-optional" : Boolean
  , "save-prefix": String
  , scope : String
  , searchopts : String
  , searchexclude: [null, String]
  , searchsort: [ "name", "-name"
                , "description", "-description"
                , "author", "-author"
                , "date", "-date"
                , "keywords", "-keywords" ]
  , shell : String
  , shrinkwrap: Boolean
  , "sign-git-tag": Boolean
  , spin: ["always", Boolean]
  , "strict-ssl": Boolean
  , tag : tag
  , tmp : path
  , unicode : Boolean
  , "unsafe-perm" : Boolean
  , usage : Boolean
  , user : [Number, String]
  , userconfig : path
  , umask: Octal
  , version : Boolean
  , versions : Boolean
  , viewer: String
  , _exit : Boolean
  }

function getLocalAddresses() {
  Object.keys(os.networkInterfaces()).map(function (nic) {
    return os.networkInterfaces()[nic].filter(function (addr) {
      return addr.family === "IPv4"
    })
    .map(function (addr) {
      return addr.address
    })
  }).reduce(function (curr, next) {
    return curr.concat(next)
  }, []).concat(undefined)
}

exports.shorthands =
  { s : ["--loglevel", "silent"]
  , d : ["--loglevel", "info"]
  , dd : ["--loglevel", "verbose"]
  , ddd : ["--loglevel", "silly"]
  , noreg : ["--no-registry"]
  , N : ["--no-registry"]
  , reg : ["--registry"]
  , "no-reg" : ["--no-registry"]
  , silent : ["--loglevel", "silent"]
  , verbose : ["--loglevel", "verbose"]
  , quiet: ["--loglevel", "warn"]
  , q: ["--loglevel", "warn"]
  , h : ["--usage"]
  , H : ["--usage"]
  , "?" : ["--usage"]
  , help : ["--usage"]
  , v : ["--version"]
  , f : ["--force"]
  , gangster : ["--force"]
  , gangsta : ["--force"]
  , desc : ["--description"]
  , "no-desc" : ["--no-description"]
  , "local" : ["--no-global"]
  , l : ["--long"]
  , m : ["--message"]
  , p : ["--parseable"]
  , porcelain : ["--parseable"]
  , g : ["--global"]
  , S : ["--save"]
  , D : ["--save-dev"]
  , E : ["--save-exact"]
  , O : ["--save-optional"]
  , y : ["--yes"]
  , n : ["--no-yes"]
  , B : ["--save-bundle"]
  , C : ["--prefix"]
  }
