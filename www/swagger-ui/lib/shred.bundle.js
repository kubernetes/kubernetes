var require = function (file, cwd) {
    var resolved = require.resolve(file, cwd || '/');
    var mod = require.modules[resolved];
    if (!mod) throw new Error(
        'Failed to resolve module ' + file + ', tried ' + resolved
    );
    var res = mod._cached ? mod._cached : mod();
    return res;
}

require.paths = [];
require.modules = {};
require.extensions = [".js",".coffee"];

require._core = {
    'assert': true,
    'events': true,
    'fs': true,
    'path': true,
    'vm': true
};

require.resolve = (function () {
    return function (x, cwd) {
        if (!cwd) cwd = '/';
        
        if (require._core[x]) return x;
        var path = require.modules.path();
        var y = cwd || '.';
        
        if (x.match(/^(?:\.\.?\/|\/)/)) {
            var m = loadAsFileSync(path.resolve(y, x))
                || loadAsDirectorySync(path.resolve(y, x));
            if (m) return m;
        }
        
        var n = loadNodeModulesSync(x, y);
        if (n) return n;
        
        throw new Error("Cannot find module '" + x + "'");
        
        function loadAsFileSync (x) {
            if (require.modules[x]) {
                return x;
            }
            
            for (var i = 0; i < require.extensions.length; i++) {
                var ext = require.extensions[i];
                if (require.modules[x + ext]) return x + ext;
            }
        }
        
        function loadAsDirectorySync (x) {
            x = x.replace(/\/+$/, '');
            var pkgfile = x + '/package.json';
            if (require.modules[pkgfile]) {
                var pkg = require.modules[pkgfile]();
                var b = pkg.browserify;
                if (typeof b === 'object' && b.main) {
                    var m = loadAsFileSync(path.resolve(x, b.main));
                    if (m) return m;
                }
                else if (typeof b === 'string') {
                    var m = loadAsFileSync(path.resolve(x, b));
                    if (m) return m;
                }
                else if (pkg.main) {
                    var m = loadAsFileSync(path.resolve(x, pkg.main));
                    if (m) return m;
                }
            }
            
            return loadAsFileSync(x + '/index');
        }
        
        function loadNodeModulesSync (x, start) {
            var dirs = nodeModulesPathsSync(start);
            for (var i = 0; i < dirs.length; i++) {
                var dir = dirs[i];
                var m = loadAsFileSync(dir + '/' + x);
                if (m) return m;
                var n = loadAsDirectorySync(dir + '/' + x);
                if (n) return n;
            }
            
            var m = loadAsFileSync(x);
            if (m) return m;
        }
        
        function nodeModulesPathsSync (start) {
            var parts;
            if (start === '/') parts = [ '' ];
            else parts = path.normalize(start).split('/');
            
            var dirs = [];
            for (var i = parts.length - 1; i >= 0; i--) {
                if (parts[i] === 'node_modules') continue;
                var dir = parts.slice(0, i + 1).join('/') + '/node_modules';
                dirs.push(dir);
            }
            
            return dirs;
        }
    };
})();

require.alias = function (from, to) {
    var path = require.modules.path();
    var res = null;
    try {
        res = require.resolve(from + '/package.json', '/');
    }
    catch (err) {
        res = require.resolve(from, '/');
    }
    var basedir = path.dirname(res);
    
    var keys = (Object.keys || function (obj) {
        var res = [];
        for (var key in obj) res.push(key)
        return res;
    })(require.modules);
    
    for (var i = 0; i < keys.length; i++) {
        var key = keys[i];
        if (key.slice(0, basedir.length + 1) === basedir + '/') {
            var f = key.slice(basedir.length);
            require.modules[to + f] = require.modules[basedir + f];
        }
        else if (key === basedir) {
            require.modules[to] = require.modules[basedir];
        }
    }
};

require.define = function (filename, fn) {
    var dirname = require._core[filename]
        ? ''
        : require.modules.path().dirname(filename)
    ;
    
    var require_ = function (file) {
        return require(file, dirname)
    };
    require_.resolve = function (name) {
        return require.resolve(name, dirname);
    };
    require_.modules = require.modules;
    require_.define = require.define;
    var module_ = { exports : {} };
    
    require.modules[filename] = function () {
        require.modules[filename]._cached = module_.exports;
        fn.call(
            module_.exports,
            require_,
            module_,
            module_.exports,
            dirname,
            filename
        );
        require.modules[filename]._cached = module_.exports;
        return module_.exports;
    };
};

if (typeof process === 'undefined') process = {};

if (!process.nextTick) process.nextTick = (function () {
    var queue = [];
    var canPost = typeof window !== 'undefined'
        && window.postMessage && window.addEventListener
    ;
    
    if (canPost) {
        window.addEventListener('message', function (ev) {
            if (ev.source === window && ev.data === 'browserify-tick') {
                ev.stopPropagation();
                if (queue.length > 0) {
                    var fn = queue.shift();
                    fn();
                }
            }
        }, true);
    }
    
    return function (fn) {
        if (canPost) {
            queue.push(fn);
            window.postMessage('browserify-tick', '*');
        }
        else setTimeout(fn, 0);
    };
})();

if (!process.title) process.title = 'browser';

if (!process.binding) process.binding = function (name) {
    if (name === 'evals') return require('vm')
    else throw new Error('No such module')
};

if (!process.cwd) process.cwd = function () { return '.' };

require.define("path", function (require, module, exports, __dirname, __filename) {
    function filter (xs, fn) {
    var res = [];
    for (var i = 0; i < xs.length; i++) {
        if (fn(xs[i], i, xs)) res.push(xs[i]);
    }
    return res;
}

// resolves . and .. elements in a path array with directory names there
// must be no slashes, empty elements, or device names (c:\) in the array
// (so also no leading and trailing slashes - it does not distinguish
// relative and absolute paths)
function normalizeArray(parts, allowAboveRoot) {
  // if the path tries to go above the root, `up` ends up > 0
  var up = 0;
  for (var i = parts.length; i >= 0; i--) {
    var last = parts[i];
    if (last == '.') {
      parts.splice(i, 1);
    } else if (last === '..') {
      parts.splice(i, 1);
      up++;
    } else if (up) {
      parts.splice(i, 1);
      up--;
    }
  }

  // if the path is allowed to go above the root, restore leading ..s
  if (allowAboveRoot) {
    for (; up--; up) {
      parts.unshift('..');
    }
  }

  return parts;
}

// Regex to split a filename into [*, dir, basename, ext]
// posix version
var splitPathRe = /^(.+\/(?!$)|\/)?((?:.+?)?(\.[^.]*)?)$/;

// path.resolve([from ...], to)
// posix version
exports.resolve = function() {
var resolvedPath = '',
    resolvedAbsolute = false;

for (var i = arguments.length; i >= -1 && !resolvedAbsolute; i--) {
  var path = (i >= 0)
      ? arguments[i]
      : process.cwd();

  // Skip empty and invalid entries
  if (typeof path !== 'string' || !path) {
    continue;
  }

  resolvedPath = path + '/' + resolvedPath;
  resolvedAbsolute = path.charAt(0) === '/';
}

// At this point the path should be resolved to a full absolute path, but
// handle relative paths to be safe (might happen when process.cwd() fails)

// Normalize the path
resolvedPath = normalizeArray(filter(resolvedPath.split('/'), function(p) {
    return !!p;
  }), !resolvedAbsolute).join('/');

  return ((resolvedAbsolute ? '/' : '') + resolvedPath) || '.';
};

// path.normalize(path)
// posix version
exports.normalize = function(path) {
var isAbsolute = path.charAt(0) === '/',
    trailingSlash = path.slice(-1) === '/';

// Normalize the path
path = normalizeArray(filter(path.split('/'), function(p) {
    return !!p;
  }), !isAbsolute).join('/');

  if (!path && !isAbsolute) {
    path = '.';
  }
  if (path && trailingSlash) {
    path += '/';
  }
  
  return (isAbsolute ? '/' : '') + path;
};


// posix version
exports.join = function() {
  var paths = Array.prototype.slice.call(arguments, 0);
  return exports.normalize(filter(paths, function(p, index) {
    return p && typeof p === 'string';
  }).join('/'));
};


exports.dirname = function(path) {
  var dir = splitPathRe.exec(path)[1] || '';
  var isWindows = false;
  if (!dir) {
    // No dirname
    return '.';
  } else if (dir.length === 1 ||
      (isWindows && dir.length <= 3 && dir.charAt(1) === ':')) {
    // It is just a slash or a drive letter with a slash
    return dir;
  } else {
    // It is a full dirname, strip trailing slash
    return dir.substring(0, dir.length - 1);
  }
};


exports.basename = function(path, ext) {
  var f = splitPathRe.exec(path)[2] || '';
  // TODO: make this comparison case-insensitive on windows?
  if (ext && f.substr(-1 * ext.length) === ext) {
    f = f.substr(0, f.length - ext.length);
  }
  return f;
};


exports.extname = function(path) {
  return splitPathRe.exec(path)[3] || '';
};

});

require.define("/shred.js", function (require, module, exports, __dirname, __filename) {
    // Shred is an HTTP client library intended to simplify the use of Node's
// built-in HTTP library. In particular, we wanted to make it easier to interact
// with HTTP-based APIs.
// 
// See the [examples](./examples.html) for more details.

// Ax is a nice logging library we wrote. You can use any logger, providing it
// has `info`, `warn`, `debug`, and `error` methods that take a string.
var Ax = require("ax")
  , CookieJarLib = require( "cookiejar" )
  , CookieJar = CookieJarLib.CookieJar
;

// Shred takes some options, including a logger and request defaults.

var Shred = function(options) {
  options = (options||{});
  this.agent = options.agent;
  this.defaults = options.defaults||{};
  this.log = options.logger||(new Ax({ level: "info" }));
  this._sharedCookieJar = new CookieJar();
  this.logCurl = options.logCurl || false;
};

// Most of the real work is done in the request and reponse classes.
 
Shred.Request = require("./shred/request");
Shred.Response = require("./shred/response");

// The `request` method kicks off a new request, instantiating a new `Request`
// object and passing along whatever default options we were given.

Shred.prototype = {
  request: function(options) {
    options.logger = this.log;
    options.logCurl = options.logCurl || this.logCurl;
    options.cookieJar = ( 'cookieJar' in options ) ? options.cookieJar : this._sharedCookieJar; // let them set cookieJar = null
    options.agent = options.agent || this.agent;
    // fill in default options
    for (var key in this.defaults) {
      if (this.defaults.hasOwnProperty(key) && !options[key]) {
        options[key] = this.defaults[key]
      }
    }
    return new Shred.Request(options);
  }
};

// Define a bunch of convenience methods so that you don't have to include
// a `method` property in your request options.

"get put post delete".split(" ").forEach(function(method) {
  Shred.prototype[method] = function(options) {
    options.method = method;
    return this.request(options);
  };
});


module.exports = Shred;

});

require.define("/node_modules/ax/package.json", function (require, module, exports, __dirname, __filename) {
    module.exports = {"main":"./lib/ax.js"}
});

require.define("/node_modules/ax/lib/ax.js", function (require, module, exports, __dirname, __filename) {
    var inspect = require("util").inspect
  , fs = require("fs")
;


// this is a quick-and-dirty logger. there are other nicer loggers out there
// but the ones i found were also somewhat involved. this one has a Ruby
// logger type interface
//
// we can easily replace this, provide the info, debug, etc. methods are the
// same. or, we can change Haiku to use a more standard node.js interface

var format = function(level,message) {
  var debug = (level=="debug"||level=="error");
  if (!message) { return message.toString(); }
  if (typeof(message) == "object") {
    if (message instanceof Error && debug) {
      return message.stack;
    } else {
      return inspect(message);
    }
  } else {
    return message.toString();
  }
};

var noOp = function(message) { return this; }
var makeLogger = function(level,fn) {
  return function(message) { 
    this.stream.write(this.format(level, message)+"\n");
    return this;
  }
};

var Logger = function(options) {
  var logger = this;
  var options = options||{};

  // Default options
  options.level = options.level || "info";
  options.timestamp = options.timestamp || true;
  options.prefix = options.prefix || "";
  logger.options = options;

  // Allows a prefix to be added to the message.
  //
  //    var logger = new Ax({ module: 'Haiku' })
  //    logger.warn('this is going to be awesome!');
  //    //=> Haiku: this is going to be awesome!
  //
  if (logger.options.module){
    logger.options.prefix = logger.options.module;
  }

  // Write to stderr or a file
  if (logger.options.file){
    logger.stream = fs.createWriteStream(logger.options.file, {"flags": "a"});
  } else {
      if(process.title === "node")
    logger.stream = process.stderr;
      else if(process.title === "browser")
    logger.stream = function () {
      // Work around weird console context issue: http://code.google.com/p/chromium/issues/detail?id=48662
      return console[logger.options.level].apply(console, arguments);
    };
  }

  switch(logger.options.level){
    case 'debug':
      ['debug', 'info', 'warn'].forEach(function (level) {
        logger[level] = Logger.writer(level);
      });
    case 'info':
      ['info', 'warn'].forEach(function (level) {
        logger[level] = Logger.writer(level);
      });
    case 'warn':
      logger.warn = Logger.writer('warn');
  }
}

// Used to define logger methods
Logger.writer = function(level){
  return function(message){
    var logger = this;

    if(process.title === "node")
  logger.stream.write(logger.format(level, message) + '\n');
    else if(process.title === "browser")
  logger.stream(logger.format(level, message) + '\n');

  };
}


Logger.prototype = {
  info: function(){},
  debug: function(){},
  warn: function(){},
  error: Logger.writer('error'),
  format: function(level, message){
    if (! message) return '';

    var logger = this
      , prefix = logger.options.prefix
      , timestamp = logger.options.timestamp ? " " + (new Date().toISOString()) : ""
    ;

    return (prefix + timestamp + ": " + message);
  }
};

module.exports = Logger;

});

require.define("util", function (require, module, exports, __dirname, __filename) {
    // todo

});

require.define("fs", function (require, module, exports, __dirname, __filename) {
    // nothing to see here... no file methods for the browser

});

require.define("/node_modules/cookiejar/package.json", function (require, module, exports, __dirname, __filename) {
    module.exports = {"main":"cookiejar.js"}
});

require.define("/node_modules/cookiejar/cookiejar.js", function (require, module, exports, __dirname, __filename) {
    exports.CookieAccessInfo=CookieAccessInfo=function CookieAccessInfo(domain,path,secure,script) {
    if(this instanceof CookieAccessInfo) {
      this.domain=domain||undefined;
      this.path=path||"/";
      this.secure=!!secure;
      this.script=!!script;
      return this;
    }
    else {
        return new CookieAccessInfo(domain,path,secure,script)    
    }
}

exports.Cookie=Cookie=function Cookie(cookiestr) {
  if(cookiestr instanceof Cookie) {
    return cookiestr;
  }
    else {
        if(this instanceof Cookie) {
          this.name = null;
          this.value = null;
          this.expiration_date = Infinity;
          this.path = "/";
          this.domain = null;
          this.secure = false; //how to define?
          this.noscript = false; //httponly
          if(cookiestr) {
            this.parse(cookiestr)
          }
          return this;
        }
        return new Cookie(cookiestr)
    }
}

Cookie.prototype.toString = function toString() {
  var str=[this.name+"="+this.value];
  if(this.expiration_date !== Infinity) {
    str.push("expires="+(new Date(this.expiration_date)).toGMTString());
  }
  if(this.domain) {
    str.push("domain="+this.domain);
  }
  if(this.path) {
    str.push("path="+this.path);
  }
  if(this.secure) {
    str.push("secure");
  }
  if(this.noscript) {
    str.push("httponly");
  }
  return str.join("; ");
}

Cookie.prototype.toValueString = function toValueString() {
  return this.name+"="+this.value;
}

var cookie_str_splitter=/[:](?=\s*[a-zA-Z0-9_\-]+\s*[=])/g
Cookie.prototype.parse = function parse(str) {
  if(this instanceof Cookie) {
      var parts=str.split(";")
      , pair=parts[0].match(/([^=]+)=((?:.|\n)*)/)
      , key=pair[1]
      , value=pair[2];
      this.name = key;
      this.value = value;
    
      for(var i=1;i<parts.length;i++) {
        pair=parts[i].match(/([^=]+)(?:=((?:.|\n)*))?/)
        , key=pair[1].trim().toLowerCase()
        , value=pair[2];
        switch(key) {
          case "httponly":
            this.noscript = true;
          break;
          case "expires":
            this.expiration_date = value
              ? Number(Date.parse(value))
              : Infinity;
          break;
          case "path":
            this.path = value
              ? value.trim()
              : "";
          break;
          case "domain":
            this.domain = value
              ? value.trim()
              : "";
          break;
          case "secure":
            this.secure = true;
          break
        }
      }
    
      return this;
  }
    return new Cookie().parse(str)
}

Cookie.prototype.matches = function matches(access_info) {
  if(this.noscript && access_info.script
  || this.secure && !access_info.secure
  || !this.collidesWith(access_info)) {
    return false
  }
  return true;
}

Cookie.prototype.collidesWith = function collidesWith(access_info) {
  if((this.path && !access_info.path) || (this.domain && !access_info.domain)) {
    return false
  }
  if(this.path && access_info.path.indexOf(this.path) !== 0) {
    return false;
  }
  if (this.domain===access_info.domain) {
    return true;
  }
  else if(this.domain && this.domain.charAt(0)===".")
  {
    var wildcard=access_info.domain.indexOf(this.domain.slice(1))
    if(wildcard===-1 || wildcard!==access_info.domain.length-this.domain.length+1) {
      return false;
    }
  }
  else if(this.domain){
    return false
  }
  return true;
}

exports.CookieJar=CookieJar=function CookieJar() {
  if(this instanceof CookieJar) {
      var cookies = {} //name: [Cookie]
    
      this.setCookie = function setCookie(cookie) {
        cookie = Cookie(cookie);
        //Delete the cookie if the set is past the current time
        var remove = cookie.expiration_date <= Date.now();
        if(cookie.name in cookies) {
          var cookies_list = cookies[cookie.name];
          for(var i=0;i<cookies_list.length;i++) {
            var collidable_cookie = cookies_list[i];
            if(collidable_cookie.collidesWith(cookie)) {
              if(remove) {
                cookies_list.splice(i,1);
                if(cookies_list.length===0) {
                  delete cookies[cookie.name]
                }
                return false;
              }
              else {
                return cookies_list[i]=cookie;
              }
            }
          }
          if(remove) {
            return false;
          }
          cookies_list.push(cookie);
          return cookie;
        }
        else if(remove){
          return false;
        }
        else {
          return cookies[cookie.name]=[cookie];
        }
      }
      //returns a cookie
      this.getCookie = function getCookie(cookie_name,access_info) {
        var cookies_list = cookies[cookie_name];
        for(var i=0;i<cookies_list.length;i++) {
          var cookie = cookies_list[i];
          if(cookie.expiration_date <= Date.now()) {
            if(cookies_list.length===0) {
              delete cookies[cookie.name]
            }
            continue;
          }
          if(cookie.matches(access_info)) {
            return cookie;
          }
        }
      }
      //returns a list of cookies
      this.getCookies = function getCookies(access_info) {
        var matches=[];
        for(var cookie_name in cookies) {
          var cookie=this.getCookie(cookie_name,access_info);
          if (cookie) {
            matches.push(cookie);
          }
        }
        matches.toString=function toString(){return matches.join(":");}
            matches.toValueString=function() {return matches.map(function(c){return c.toValueString();}).join(';');}
        return matches;
      }
    
      return this;
  }
    return new CookieJar()
}


//returns list of cookies that were set correctly
CookieJar.prototype.setCookies = function setCookies(cookies) {
  cookies=Array.isArray(cookies)
    ?cookies
    :cookies.split(cookie_str_splitter);
  var successful=[]
  for(var i=0;i<cookies.length;i++) {
    var cookie = Cookie(cookies[i]);
    if(this.setCookie(cookie)) {
      successful.push(cookie);
    }
  }
  return successful;
}

});

require.define("/shred/request.js", function (require, module, exports, __dirname, __filename) {
    // The request object encapsulates a request, creating a Node.js HTTP request and
// then handling the response.

var HTTP = require("http")
  , HTTPS = require("https")
  , parseUri = require("./parseUri")
  , Emitter = require('events').EventEmitter
  , sprintf = require("sprintf").sprintf
  , Response = require("./response")
  , HeaderMixins = require("./mixins/headers")
  , Content = require("./content")
;

var STATUS_CODES = HTTP.STATUS_CODES || {
    100 : 'Continue',
    101 : 'Switching Protocols',
    102 : 'Processing', // RFC 2518, obsoleted by RFC 4918
    200 : 'OK',
    201 : 'Created',
    202 : 'Accepted',
    203 : 'Non-Authoritative Information',
    204 : 'No Content',
    205 : 'Reset Content',
    206 : 'Partial Content',
    207 : 'Multi-Status', // RFC 4918
    300 : 'Multiple Choices',
    301 : 'Moved Permanently',
    302 : 'Moved Temporarily',
    303 : 'See Other',
    304 : 'Not Modified',
    305 : 'Use Proxy',
    307 : 'Temporary Redirect',
    400 : 'Bad Request',
    401 : 'Unauthorized',
    402 : 'Payment Required',
    403 : 'Forbidden',
    404 : 'Not Found',
    405 : 'Method Not Allowed',
    406 : 'Not Acceptable',
    407 : 'Proxy Authentication Required',
    408 : 'Request Time-out',
    409 : 'Conflict',
    410 : 'Gone',
    411 : 'Length Required',
    412 : 'Precondition Failed',
    413 : 'Request Entity Too Large',
    414 : 'Request-URI Too Large',
    415 : 'Unsupported Media Type',
    416 : 'Requested Range Not Satisfiable',
    417 : 'Expectation Failed',
    418 : 'I\'m a teapot', // RFC 2324
    422 : 'Unprocessable Entity', // RFC 4918
    423 : 'Locked', // RFC 4918
    424 : 'Failed Dependency', // RFC 4918
    425 : 'Unordered Collection', // RFC 4918
    426 : 'Upgrade Required', // RFC 2817
    500 : 'Internal Server Error',
    501 : 'Not Implemented',
    502 : 'Bad Gateway',
    503 : 'Service Unavailable',
    504 : 'Gateway Time-out',
    505 : 'HTTP Version not supported',
    506 : 'Variant Also Negotiates', // RFC 2295
    507 : 'Insufficient Storage', // RFC 4918
    509 : 'Bandwidth Limit Exceeded',
    510 : 'Not Extended' // RFC 2774
};

// The Shred object itself constructs the `Request` object. You should rarely
// need to do this directly.

var Request = function(options) {
  this.log = options.logger;
  this.cookieJar = options.cookieJar;
  this.encoding = options.encoding;
  this.logCurl = options.logCurl;
  processOptions(this,options||{});
  createRequest(this);
};

// A `Request` has a number of properties, many of which help with details like
// URL parsing or defaulting the port for the request.

Object.defineProperties(Request.prototype, {

// - **url**. You can set the `url` property with a valid URL string and all the
//   URL-related properties (host, port, etc.) will be automatically set on the
//   request object.

  url: {
    get: function() {
      if (!this.scheme) { return null; }
      return sprintf("%s://%s:%s%s",
          this.scheme, this.host, this.port,
          (this.proxy ? "/" : this.path) +
          (this.query ? ("?" + this.query) : ""));
    },
    set: function(_url) {
      _url = parseUri(_url);
      this.scheme = _url.protocol;
      this.host = _url.host;
      this.port = _url.port;
      this.path = _url.path;
      this.query = _url.query;
      return this;
    },
    enumerable: true
  },

// - **headers**. Returns a hash representing the request headers. You can't set
//   this directly, only get it. You can add or modify headers by using the
//   `setHeader` or `setHeaders` method. This ensures that the headers are
//   normalized - that is, you don't accidentally send `Content-Type` and
//   `content-type` headers. Keep in mind that if you modify the returned hash,
//   it will *not* modify the request headers.

  headers: {
    get: function() {
      return this.getHeaders();
    },
    enumerable: true
  },

// - **port**. Unless you set the `port` explicitly or include it in the URL, it
//   will default based on the scheme.

  port: {
    get: function() {
      if (!this._port) {
        switch(this.scheme) {
          case "https": return this._port = 443;
          case "http":
          default: return this._port = 80;
        }
      }
      return this._port;
    },
    set: function(value) { this._port = value; return this; },
    enumerable: true
  },

// - **method**. The request method - `get`, `put`, `post`, etc. that will be
//   used to make the request. Defaults to `get`.

  method: {
    get: function() {
      return this._method = (this._method||"GET");
    },
    set: function(value) {
      this._method = value; return this;
    },
    enumerable: true
  },

// - **query**. Can be set either with a query string or a hash (object). Get
//   will always return a properly escaped query string or null if there is no
//   query component for the request.

  query: {
    get: function() {return this._query;},
    set: function(value) {
      var stringify = function (hash) {
        var query = "";
        for (var key in hash) {
          query += encodeURIComponent(key) + '=' + encodeURIComponent(hash[key]) + '&';
        }
        // Remove the last '&'
        query = query.slice(0, -1);
        return query;
      }

      if (value) {
        if (typeof value === 'object') {
          value = stringify(value);
        }
        this._query = value;
      } else {
        this._query = "";
      }
      return this;
    },
    enumerable: true
  },

// - **parameters**. This will return the query parameters in the form of a hash
//   (object).

  parameters: {
    get: function() { return QueryString.parse(this._query||""); },
    enumerable: true
  },

// - **content**. (Aliased as `body`.) Set this to add a content entity to the
//   request. Attempts to use the `content-type` header to determine what to do
//   with the content value. Get this to get back a [`Content`
//   object](./content.html).

  body: {
    get: function() { return this._body; },
    set: function(value) {
      this._body = new Content({
        data: value,
        type: this.getHeader("Content-Type")
      });
      this.setHeader("Content-Type",this.content.type);
      this.setHeader("Content-Length",this.content.length);
      return this;
    },
    enumerable: true
  },

// - **timeout**. Used to determine how long to wait for a response. Does not
//   distinguish between connect timeouts versus request timeouts. Set either in
//   milliseconds or with an object with temporal attributes (hours, minutes,
//   seconds) and convert it into milliseconds. Get will always return
//   milliseconds.

  timeout: {
    get: function() { return this._timeout; }, // in milliseconds
    set: function(timeout) {
      var request = this
        , milliseconds = 0;
      ;
      if (!timeout) return this;
      if (typeof timeout==="number") { milliseconds = timeout; }
      else {
        milliseconds = (timeout.milliseconds||0) +
          (1000 * ((timeout.seconds||0) +
              (60 * ((timeout.minutes||0) +
                (60 * (timeout.hours||0))))));
      }
      this._timeout = milliseconds;
      return this;
    },
    enumerable: true
  }
});

// Alias `body` property to `content`. Since the [content object](./content.html)
// has a `body` attribute, it's preferable to use `content` since you can then
// access the raw content data using `content.body`.

Object.defineProperty(Request.prototype,"content",
    Object.getOwnPropertyDescriptor(Request.prototype, "body"));

// The `Request` object can be pretty overwhelming to view using the built-in
// Node.js inspect method. We want to make it a bit more manageable. This
// probably goes [too far in the other
// direction](https://github.com/spire-io/shred/issues/2).

Request.prototype.inspect = function () {
  var request = this;
  var headers = this.format_headers();
  var summary = ["<Shred Request> ", request.method.toUpperCase(),
      request.url].join(" ")
  return [ summary, "- Headers:", headers].join("\n");
};

Request.prototype.format_headers = function () {
  var array = []
  var headers = this._headers
  for (var key in headers) {
    if (headers.hasOwnProperty(key)) {
      var value = headers[key]
      array.push("\t" + key + ": " + value);
    }
  }
  return array.join("\n");
};

// Allow chainable 'on's:  shred.get({ ... }).on( ... ).  You can pass in a
// single function, a pair (event, function), or a hash:
// { event: function, event: function }
Request.prototype.on = function (eventOrHash, listener) {
  var emitter = this.emitter;
  // Pass in a single argument as a function then make it the default response handler
  if (arguments.length === 1 && typeof(eventOrHash) === 'function') {
    emitter.on('response', eventOrHash);
  } else if (arguments.length === 1 && typeof(eventOrHash) === 'object') {
    for (var key in eventOrHash) {
      if (eventOrHash.hasOwnProperty(key)) {
        emitter.on(key, eventOrHash[key]);
      }
    }
  } else {
    emitter.on(eventOrHash, listener);
  }
  return this;
};

// Add in the header methods. Again, these ensure we don't get the same header
// multiple times with different case conventions.
HeaderMixins.gettersAndSetters(Request);

// `processOptions` is called from the constructor to handle all the work
// associated with making sure we do our best to ensure we have a valid request.

var processOptions = function(request,options) {

  request.log.debug("Processing request options ..");

  // We'll use `request.emitter` to manage the `on` event handlers.
  request.emitter = (new Emitter);

  request.agent = options.agent;

  // Set up the handlers ...
  if (options.on) {
    for (var key in options.on) {
      if (options.on.hasOwnProperty(key)) {
        request.emitter.on(key, options.on[key]);
      }
    }
  }

  // Make sure we were give a URL or a host
  if (!options.url && !options.host) {
    request.emitter.emit("request_error",
        new Error("No url or url options (host, port, etc.)"));
    return;
  }

  // Allow for the [use of a proxy](http://www.jmarshall.com/easy/http/#proxies).

  if (options.url) {
    if (options.proxy) {
      request.url = options.proxy;
      request.path = options.url;
    } else {
      request.url = options.url;
    }
  }

  // Set the remaining options.
  request.query = options.query||options.parameters||request.query ;
  request.method = options.method;
  request.setHeader("user-agent",options.agent||"Shred");
  request.setHeaders(options.headers);

  if (request.cookieJar) {
    var cookies = request.cookieJar.getCookies( CookieAccessInfo( request.host, request.path ) );
    if (cookies.length) {
      var cookieString = request.getHeader('cookie')||'';
      for (var cookieIndex = 0; cookieIndex < cookies.length; ++cookieIndex) {
          if ( cookieString.length && cookieString[ cookieString.length - 1 ] != ';' )
          {
              cookieString += ';';
          }
          cookieString += cookies[ cookieIndex ].name + '=' + cookies[ cookieIndex ].value + ';';
      }
      request.setHeader("cookie", cookieString);
    }
  }
  
  // The content entity can be set either using the `body` or `content` attributes.
  if (options.body||options.content) {
    request.content = options.body||options.content;
  }
  request.timeout = options.timeout;

};

// `createRequest` is also called by the constructor, after `processOptions`.
// This actually makes the request and processes the response, so `createRequest`
// is a bit of a misnomer.

var createRequest = function(request) {
  var timeout ;

  request.log.debug("Creating request ..");
  request.log.debug(request);

  var reqParams = {
    host: request.host,
    port: request.port,
    method: request.method,
    path: request.path + (request.query ? '?'+request.query : ""),
    headers: request.getHeaders(),
    // Node's HTTP/S modules will ignore this, but we are using the
    // browserify-http module in the browser for both HTTP and HTTPS, and this
    // is how you differentiate the two.
    scheme: request.scheme,
    // Use a provided agent.  'Undefined' is the default, which uses a global
    // agent.
    agent: request.agent
  };

  if (request.logCurl) {
    logCurl(request);
  }

  var http = request.scheme == "http" ? HTTP : HTTPS;

  // Set up the real request using the selected library. The request won't be
  // sent until we call `.end()`.
  request._raw = http.request(reqParams, function(response) {
    request.log.debug("Received response ..");

    // We haven't timed out and we have a response, so make sure we clear the
    // timeout so it doesn't fire while we're processing the response.
    clearTimeout(timeout);

    // Construct a Shred `Response` object from the response. This will stream
    // the response, thus the need for the callback. We can access the response
    // entity safely once we're in the callback.
    response = new Response(response, request, function(response) {

      // Set up some event magic. The precedence is given first to
      // status-specific handlers, then to responses for a given event, and then
      // finally to the more general `response` handler. In the last case, we
      // need to first make sure we're not dealing with a a redirect.
      var emit = function(event) {
        var emitter = request.emitter;
        var textStatus = STATUS_CODES[response.status] ? STATUS_CODES[response.status].toLowerCase() : null;
        if (emitter.listeners(response.status).length > 0 || emitter.listeners(textStatus).length > 0) {
          emitter.emit(response.status, response);
          emitter.emit(textStatus, response);
        } else {
          if (emitter.listeners(event).length>0) {
            emitter.emit(event, response);
          } else if (!response.isRedirect) {
            emitter.emit("response", response);
            //console.warn("Request has no event listener for status code " + response.status);
          }
        }
      };

      // Next, check for a redirect. We simply repeat the request with the URL
      // given in the `Location` header. We fire a `redirect` event.
      if (response.isRedirect) {
        request.log.debug("Redirecting to "
            + response.getHeader("Location"));
        request.url = response.getHeader("Location");
        emit("redirect");
        createRequest(request);

      // Okay, it's not a redirect. Is it an error of some kind?
      } else if (response.isError) {
        emit("error");
      } else {
      // It looks like we're good shape. Trigger the `success` event.
        emit("success");
      }
    });
  });

  // We're still setting up the request. Next, we're going to handle error cases
  // where we have no response. We don't emit an error event because that event
  // takes a response. We don't response handlers to have to check for a null
  // value. However, we [should introduce a different event
  // type](https://github.com/spire-io/shred/issues/3) for this type of error.
  request._raw.on("error", function(error) {
    request.emitter.emit("request_error", error);
  });

  request._raw.on("socket", function(socket) {
    request.emitter.emit("socket", socket);
  });

  // TCP timeouts should also trigger the "response_error" event.
  request._raw.on('socket', function () {
    request._raw.socket.on('timeout', function () {
      // This should trigger the "error" event on the raw request, which will
      // trigger the "response_error" on the shred request.
      request._raw.abort();
    });
  });


  // We're almost there. Next, we need to write the request entity to the
  // underlying request object.
  if (request.content) {
    request.log.debug("Streaming body: '" +
        request.content.data.slice(0,59) + "' ... ");
    request._raw.write(request.content.data);
  }

  // Finally, we need to set up the timeout. We do this last so that we don't
  // start the clock ticking until the last possible moment.
  if (request.timeout) {
    timeout = setTimeout(function() {
      request.log.debug("Timeout fired, aborting request ...");
      request._raw.abort();
      request.emitter.emit("timeout", request);
    },request.timeout);
  }

  // The `.end()` method will cause the request to fire. Technically, it might
  // have already sent the headers and body.
  request.log.debug("Sending request ...");
  request._raw.end();
};

// Logs the curl command for the request.
var logCurl = function (req) {
  var headers = req.getHeaders();
  var headerString = "";

  for (var key in headers) {
    headerString += '-H "' + key + ": " + headers[key] + '" ';
  }

  var bodyString = ""

  if (req.content) {
    bodyString += "-d '" + req.content.body + "' ";
  }

  var query = req.query ? '?' + req.query : "";

  console.log("curl " +
    "-X " + req.method.toUpperCase() + " " +
    req.scheme + "://" + req.host + ":" + req.port + req.path + query + " " +
    headerString +
    bodyString
  );
};


module.exports = Request;

});

require.define("http", function (require, module, exports, __dirname, __filename) {
    // todo

});

require.define("https", function (require, module, exports, __dirname, __filename) {
    // todo

});

require.define("/shred/parseUri.js", function (require, module, exports, __dirname, __filename) {
    // parseUri 1.2.2
// (c) Steven Levithan <stevenlevithan.com>
// MIT License

function parseUri (str) {
  var o   = parseUri.options,
    m   = o.parser[o.strictMode ? "strict" : "loose"].exec(str),
    uri = {},
    i   = 14;

  while (i--) uri[o.key[i]] = m[i] || "";

  uri[o.q.name] = {};
  uri[o.key[12]].replace(o.q.parser, function ($0, $1, $2) {
    if ($1) uri[o.q.name][$1] = $2;
  });

  return uri;
};

parseUri.options = {
  strictMode: false,
  key: ["source","protocol","authority","userInfo","user","password","host","port","relative","path","directory","file","query","anchor"],
  q:   {
    name:   "queryKey",
    parser: /(?:^|&)([^&=]*)=?([^&]*)/g
  },
  parser: {
    strict: /^(?:([^:\/?#]+):)?(?:\/\/((?:(([^:@]*)(?::([^:@]*))?)?@)?([^:\/?#]*)(?::(\d*))?))?((((?:[^?#\/]*\/)*)([^?#]*))(?:\?([^#]*))?(?:#(.*))?)/,
    loose:  /^(?:(?![^:@]+:[^:@\/]*@)([^:\/?#.]+):)?(?:\/\/)?((?:(([^:@]*)(?::([^:@]*))?)?@)?([^:\/?#]*)(?::(\d*))?)(((\/(?:[^?#](?![^?#\/]*\.[^?#\/.]+(?:[?#]|$)))*\/?)?([^?#\/]*))(?:\?([^#]*))?(?:#(.*))?)/
  }
};

module.exports = parseUri;

});

require.define("events", function (require, module, exports, __dirname, __filename) {
    if (!process.EventEmitter) process.EventEmitter = function () {};

var EventEmitter = exports.EventEmitter = process.EventEmitter;
var isArray = typeof Array.isArray === 'function'
    ? Array.isArray
    : function (xs) {
        return Object.toString.call(xs) === '[object Array]'
    }
;

// By default EventEmitters will print a warning if more than
// 10 listeners are added to it. This is a useful default which
// helps finding memory leaks.
//
// Obviously not all Emitters should be limited to 10. This function allows
// that to be increased. Set to zero for unlimited.
var defaultMaxListeners = 10;
EventEmitter.prototype.setMaxListeners = function(n) {
  if (!this._events) this._events = {};
  this._events.maxListeners = n;
};


EventEmitter.prototype.emit = function(type) {
  // If there is no 'error' event listener then throw.
  if (type === 'error') {
    if (!this._events || !this._events.error ||
        (isArray(this._events.error) && !this._events.error.length))
    {
      if (arguments[1] instanceof Error) {
        throw arguments[1]; // Unhandled 'error' event
      } else {
        throw new Error("Uncaught, unspecified 'error' event.");
      }
      return false;
    }
  }

  if (!this._events) return false;
  var handler = this._events[type];
  if (!handler) return false;

  if (typeof handler == 'function') {
    switch (arguments.length) {
      // fast cases
      case 1:
        handler.call(this);
        break;
      case 2:
        handler.call(this, arguments[1]);
        break;
      case 3:
        handler.call(this, arguments[1], arguments[2]);
        break;
      // slower
      default:
        var args = Array.prototype.slice.call(arguments, 1);
        handler.apply(this, args);
    }
    return true;

  } else if (isArray(handler)) {
    var args = Array.prototype.slice.call(arguments, 1);

    var listeners = handler.slice();
    for (var i = 0, l = listeners.length; i < l; i++) {
      listeners[i].apply(this, args);
    }
    return true;

  } else {
    return false;
  }
};

// EventEmitter is defined in src/node_events.cc
// EventEmitter.prototype.emit() is also defined there.
EventEmitter.prototype.addListener = function(type, listener) {
  if ('function' !== typeof listener) {
    throw new Error('addListener only takes instances of Function');
  }

  if (!this._events) this._events = {};

  // To avoid recursion in the case that type == "newListeners"! Before
  // adding it to the listeners, first emit "newListeners".
  this.emit('newListener', type, listener);

  if (!this._events[type]) {
    // Optimize the case of one listener. Don't need the extra array object.
    this._events[type] = listener;
  } else if (isArray(this._events[type])) {

    // Check for listener leak
    if (!this._events[type].warned) {
      var m;
      if (this._events.maxListeners !== undefined) {
        m = this._events.maxListeners;
      } else {
        m = defaultMaxListeners;
      }

      if (m && m > 0 && this._events[type].length > m) {
        this._events[type].warned = true;
        console.error('(node) warning: possible EventEmitter memory ' +
                      'leak detected. %d listeners added. ' +
                      'Use emitter.setMaxListeners() to increase limit.',
                      this._events[type].length);
        console.trace();
      }
    }

    // If we've already got an array, just append.
    this._events[type].push(listener);
  } else {
    // Adding the second element, need to change to array.
    this._events[type] = [this._events[type], listener];
  }

  return this;
};

EventEmitter.prototype.on = EventEmitter.prototype.addListener;

EventEmitter.prototype.once = function(type, listener) {
  var self = this;
  self.on(type, function g() {
    self.removeListener(type, g);
    listener.apply(this, arguments);
  });

  return this;
};

EventEmitter.prototype.removeListener = function(type, listener) {
  if ('function' !== typeof listener) {
    throw new Error('removeListener only takes instances of Function');
  }

  // does not use listeners(), so no side effect of creating _events[type]
  if (!this._events || !this._events[type]) return this;

  var list = this._events[type];

  if (isArray(list)) {
    var i = list.indexOf(listener);
    if (i < 0) return this;
    list.splice(i, 1);
    if (list.length == 0)
      delete this._events[type];
  } else if (this._events[type] === listener) {
    delete this._events[type];
  }

  return this;
};

EventEmitter.prototype.removeAllListeners = function(type) {
  // does not use listeners(), so no side effect of creating _events[type]
  if (type && this._events && this._events[type]) this._events[type] = null;
  return this;
};

EventEmitter.prototype.listeners = function(type) {
  if (!this._events) this._events = {};
  if (!this._events[type]) this._events[type] = [];
  if (!isArray(this._events[type])) {
    this._events[type] = [this._events[type]];
  }
  return this._events[type];
};

});

require.define("/node_modules/sprintf/package.json", function (require, module, exports, __dirname, __filename) {
    module.exports = {"main":"./lib/sprintf"}
});

require.define("/node_modules/sprintf/lib/sprintf.js", function (require, module, exports, __dirname, __filename) {
    /**
sprintf() for JavaScript 0.7-beta1
http://www.diveintojavascript.com/projects/javascript-sprintf

Copyright (c) Alexandru Marasteanu <alexaholic [at) gmail (dot] com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of sprintf() for JavaScript nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Alexandru Marasteanu BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Changelog:
2010.11.07 - 0.7-beta1-node
  - converted it to a node.js compatible module

2010.09.06 - 0.7-beta1
  - features: vsprintf, support for named placeholders
  - enhancements: format cache, reduced global namespace pollution

2010.05.22 - 0.6:
 - reverted to 0.4 and fixed the bug regarding the sign of the number 0
 Note:
 Thanks to Raphael Pigulla <raph (at] n3rd [dot) org> (http://www.n3rd.org/)
 who warned me about a bug in 0.5, I discovered that the last update was
 a regress. I appologize for that.

2010.05.09 - 0.5:
 - bug fix: 0 is now preceeded with a + sign
 - bug fix: the sign was not at the right position on padded results (Kamal Abdali)
 - switched from GPL to BSD license

2007.10.21 - 0.4:
 - unit test and patch (David Baird)

2007.09.17 - 0.3:
 - bug fix: no longer throws exception on empty paramenters (Hans Pufal)

2007.09.11 - 0.2:
 - feature: added argument swapping

2007.04.03 - 0.1:
 - initial release
**/

var sprintf = (function() {
  function get_type(variable) {
    return Object.prototype.toString.call(variable).slice(8, -1).toLowerCase();
  }
  function str_repeat(input, multiplier) {
    for (var output = []; multiplier > 0; output[--multiplier] = input) {/* do nothing */}
    return output.join('');
  }

  var str_format = function() {
    if (!str_format.cache.hasOwnProperty(arguments[0])) {
      str_format.cache[arguments[0]] = str_format.parse(arguments[0]);
    }
    return str_format.format.call(null, str_format.cache[arguments[0]], arguments);
  };

  str_format.format = function(parse_tree, argv) {
    var cursor = 1, tree_length = parse_tree.length, node_type = '', arg, output = [], i, k, match, pad, pad_character, pad_length;
    for (i = 0; i < tree_length; i++) {
      node_type = get_type(parse_tree[i]);
      if (node_type === 'string') {
        output.push(parse_tree[i]);
      }
      else if (node_type === 'array') {
        match = parse_tree[i]; // convenience purposes only
        if (match[2]) { // keyword argument
          arg = argv[cursor];
          for (k = 0; k < match[2].length; k++) {
            if (!arg.hasOwnProperty(match[2][k])) {
              throw(sprintf('[sprintf] property "%s" does not exist', match[2][k]));
            }
            arg = arg[match[2][k]];
          }
        }
        else if (match[1]) { // positional argument (explicit)
          arg = argv[match[1]];
        }
        else { // positional argument (implicit)
          arg = argv[cursor++];
        }

        if (/[^s]/.test(match[8]) && (get_type(arg) != 'number')) {
          throw(sprintf('[sprintf] expecting number but found %s', get_type(arg)));
        }
        switch (match[8]) {
          case 'b': arg = arg.toString(2); break;
          case 'c': arg = String.fromCharCode(arg); break;
          case 'd': arg = parseInt(arg, 10); break;
          case 'e': arg = match[7] ? arg.toExponential(match[7]) : arg.toExponential(); break;
          case 'f': arg = match[7] ? parseFloat(arg).toFixed(match[7]) : parseFloat(arg); break;
          case 'o': arg = arg.toString(8); break;
          case 's': arg = ((arg = String(arg)) && match[7] ? arg.substring(0, match[7]) : arg); break;
          case 'u': arg = Math.abs(arg); break;
          case 'x': arg = arg.toString(16); break;
          case 'X': arg = arg.toString(16).toUpperCase(); break;
        }
        arg = (/[def]/.test(match[8]) && match[3] && arg >= 0 ? '+'+ arg : arg);
        pad_character = match[4] ? match[4] == '0' ? '0' : match[4].charAt(1) : ' ';
        pad_length = match[6] - String(arg).length;
        pad = match[6] ? str_repeat(pad_character, pad_length) : '';
        output.push(match[5] ? arg + pad : pad + arg);
      }
    }
    return output.join('');
  };

  str_format.cache = {};

  str_format.parse = function(fmt) {
    var _fmt = fmt, match = [], parse_tree = [], arg_names = 0;
    while (_fmt) {
      if ((match = /^[^\x25]+/.exec(_fmt)) !== null) {
        parse_tree.push(match[0]);
      }
      else if ((match = /^\x25{2}/.exec(_fmt)) !== null) {
        parse_tree.push('%');
      }
      else if ((match = /^\x25(?:([1-9]\d*)\$|\(([^\)]+)\))?(\+)?(0|'[^$])?(-)?(\d+)?(?:\.(\d+))?([b-fosuxX])/.exec(_fmt)) !== null) {
        if (match[2]) {
          arg_names |= 1;
          var field_list = [], replacement_field = match[2], field_match = [];
          if ((field_match = /^([a-z_][a-z_\d]*)/i.exec(replacement_field)) !== null) {
            field_list.push(field_match[1]);
            while ((replacement_field = replacement_field.substring(field_match[0].length)) !== '') {
              if ((field_match = /^\.([a-z_][a-z_\d]*)/i.exec(replacement_field)) !== null) {
                field_list.push(field_match[1]);
              }
              else if ((field_match = /^\[(\d+)\]/.exec(replacement_field)) !== null) {
                field_list.push(field_match[1]);
              }
              else {
                throw('[sprintf] huh?');
              }
            }
          }
          else {
            throw('[sprintf] huh?');
          }
          match[2] = field_list;
        }
        else {
          arg_names |= 2;
        }
        if (arg_names === 3) {
          throw('[sprintf] mixing positional and named placeholders is not (yet) supported');
        }
        parse_tree.push(match);
      }
      else {
        throw('[sprintf] huh?');
      }
      _fmt = _fmt.substring(match[0].length);
    }
    return parse_tree;
  };

  return str_format;
})();

var vsprintf = function(fmt, argv) {
  argv.unshift(fmt);
  return sprintf.apply(null, argv);
};

exports.sprintf = sprintf;
exports.vsprintf = vsprintf;
});

require.define("/shred/response.js", function (require, module, exports, __dirname, __filename) {
    // The `Response object` encapsulates a Node.js HTTP response.

var Content = require("./content")
  , HeaderMixins = require("./mixins/headers")
  , CookieJarLib = require( "cookiejar" )
  , Cookie = CookieJarLib.Cookie
;

// Browser doesn't have zlib.
var zlib = null;
try {
  zlib = require('zlib');
} catch (e) {
  console.warn("no zlib library");
}

// Iconv doesn't work in browser
var Iconv = null;
try {
  Iconv = require('iconv-lite');
} catch (e) {
  console.warn("no iconv library");
}

// Construct a `Response` object. You should never have to do this directly. The
// `Request` object handles this, getting the raw response object and passing it
// in here, along with the request. The callback allows us to stream the response
// and then use the callback to let the request know when it's ready.
var Response = function(raw, request, callback) { 
  var response = this;
  this._raw = raw;

  // The `._setHeaders` method is "private"; you can't otherwise set headers on
  // the response.
  this._setHeaders.call(this,raw.headers);
  
  // store any cookies
  if (request.cookieJar && this.getHeader('set-cookie')) {
    var cookieStrings = this.getHeader('set-cookie');
    var cookieObjs = []
      , cookie;

    for (var i = 0; i < cookieStrings.length; i++) {
      var cookieString = cookieStrings[i];
      if (!cookieString) {
        continue;
      }

      if (!cookieString.match(/domain\=/i)) {
        cookieString += '; domain=' + request.host;
      }

      if (!cookieString.match(/path\=/i)) {
        cookieString += '; path=' + request.path;
      }

      try {
        cookie = new Cookie(cookieString);
        if (cookie) {
          cookieObjs.push(cookie);
        }
      } catch (e) {
        console.warn("Tried to set bad cookie: " + cookieString);
      }
    }

    request.cookieJar.setCookies(cookieObjs);
  }

  this.request = request;
  this.client = request.client;
  this.log = this.request.log;

  // Stream the response content entity and fire the callback when we're done.
  // Store the incoming data in a array of Buffers which we concatinate into one
  // buffer at the end.  We need to use buffers instead of strings here in order
  // to preserve binary data.
  var chunkBuffers = [];
  var dataLength = 0;
  raw.on("data", function(chunk) {
    chunkBuffers.push(chunk);
    dataLength += chunk.length;
  });
  raw.on("end", function() {
    var body;
    if (typeof Buffer === 'undefined') {
      // Just concatinate into a string
      body = chunkBuffers.join('');
    } else {
      // Initialize new buffer and add the chunks one-at-a-time.
      body = new Buffer(dataLength);
      for (var i = 0, pos = 0; i < chunkBuffers.length; i++) {
        chunkBuffers[i].copy(body, pos);
        pos += chunkBuffers[i].length;
      }
    }

    var setBodyAndFinish = function (body) {
      response._body = new Content({ 
        body: body,
        type: response.getHeader("Content-Type")
      });
      callback(response);
    }

    if (zlib && response.getHeader("Content-Encoding") === 'gzip'){
      zlib.gunzip(body, function (err, gunzippedBody) {
        if (Iconv && response.request.encoding){
          body = Iconv.fromEncoding(gunzippedBody,response.request.encoding);
        } else {
          body = gunzippedBody.toString();
        }
        setBodyAndFinish(body);
      })
    }
    else{
       if (response.request.encoding){
            body = Iconv.fromEncoding(body,response.request.encoding);
        }        
      setBodyAndFinish(body);
    }
  });
};

// The `Response` object can be pretty overwhelming to view using the built-in
// Node.js inspect method. We want to make it a bit more manageable. This
// probably goes [too far in the other
// direction](https://github.com/spire-io/shred/issues/2).

Response.prototype = {
  inspect: function() {
    var response = this;
    var headers = this.format_headers();
    var summary = ["<Shred Response> ", response.status].join(" ")
    return [ summary, "- Headers:", headers].join("\n");
  },
  format_headers: function () {
    var array = []
    var headers = this._headers
    for (var key in headers) {
      if (headers.hasOwnProperty(key)) {
        var value = headers[key]
        array.push("\t" + key + ": " + value);
      }
    }
    return array.join("\n");
  }
};

// `Response` object properties, all of which are read-only:
Object.defineProperties(Response.prototype, {
  
// - **status**. The HTTP status code for the response. 
  status: {
    get: function() { return this._raw.statusCode; },
    enumerable: true
  },

// - **content**. The HTTP content entity, if any. Provided as a [content
//   object](./content.html), which will attempt to convert the entity based upon
//   the `content-type` header. The converted value is available as
//   `content.data`. The original raw content entity is available as
//   `content.body`.
  body: {
    get: function() { return this._body; }
  },
  content: {
    get: function() { return this.body; },
    enumerable: true
  },

// - **isRedirect**. Is the response a redirect? These are responses with 3xx
//   status and a `Location` header.
  isRedirect: {
    get: function() {
      return (this.status>299
          &&this.status<400
          &&this.getHeader("Location"));
    },
    enumerable: true
  },

// - **isError**. Is the response an error? These are responses with status of
//   400 or greater.
  isError: {
    get: function() {
      return (this.status === 0 || this.status > 399)
    },
    enumerable: true
  }
});

// Add in the [getters for accessing the normalized headers](./headers.js).
HeaderMixins.getters(Response);
HeaderMixins.privateSetters(Response);

// Work around Mozilla bug #608735 [https://bugzil.la/608735], which causes
// getAllResponseHeaders() to return {} if the response is a CORS request.
// xhr.getHeader still works correctly.
var getHeader = Response.prototype.getHeader;
Response.prototype.getHeader = function (name) {
  return (getHeader.call(this,name) ||
    (typeof this._raw.getHeader === 'function' && this._raw.getHeader(name)));
};

module.exports = Response;

});

require.define("/shred/content.js", function (require, module, exports, __dirname, __filename) {
    
// The purpose of the `Content` object is to abstract away the data conversions
// to and from raw content entities as strings. For example, you want to be able
// to pass in a Javascript object and have it be automatically converted into a
// JSON string if the `content-type` is set to a JSON-based media type.
// Conversely, you want to be able to transparently get back a Javascript object
// in the response if the `content-type` is a JSON-based media-type.

// One limitation of the current implementation is that it [assumes the `charset` is UTF-8](https://github.com/spire-io/shred/issues/5).

// The `Content` constructor takes an options object, which *must* have either a
// `body` or `data` property and *may* have a `type` property indicating the
// media type. If there is no `type` attribute, a default will be inferred.
var Content = function(options) {
  this.body = options.body;
  this.data = options.data;
  this.type = options.type;
};

Content.prototype = {
  // Treat `toString()` as asking for the `content.body`. That is, the raw content entity.
  //
  //     toString: function() { return this.body; }
  //
  // Commented out, but I've forgotten why. :/
};


// `Content` objects have the following attributes:
Object.defineProperties(Content.prototype,{
  
// - **type**. Typically accessed as `content.type`, reflects the `content-type`
//   header associated with the request or response. If not passed as an options
//   to the constructor or set explicitly, it will infer the type the `data`
//   attribute, if possible, and, failing that, will default to `text/plain`.
  type: {
    get: function() {
      if (this._type) {
        return this._type;
      } else {
        if (this._data) {
          switch(typeof this._data) {
            case "string": return "text/plain";
            case "object": return "application/json";
          }
        }
      }
      return "text/plain";
    },
    set: function(value) {
      this._type = value;
      return this;
    },
    enumerable: true
  },

// - **data**. Typically accessed as `content.data`, reflects the content entity
//   converted into Javascript data. This can be a string, if the `type` is, say,
//   `text/plain`, but can also be a Javascript object. The conversion applied is
//   based on the `processor` attribute. The `data` attribute can also be set
//   directly, in which case the conversion will be done the other way, to infer
//   the `body` attribute.
  data: {
    get: function() {
      if (this._body) {
        return this.processor.parser(this._body);
      } else {
        return this._data;
      }
    },
    set: function(data) {
      if (this._body&&data) Errors.setDataWithBody(this);
      this._data = data;
      return this;
    },
    enumerable: true
  },

// - **body**. Typically accessed as `content.body`, reflects the content entity
//   as a UTF-8 string. It is the mirror of the `data` attribute. If you set the
//   `data` attribute, the `body` attribute will be inferred and vice-versa. If
//   you attempt to set both, an exception is raised.
  body: {
    get: function() {
      if (this._data) {
        return this.processor.stringify(this._data);
      } else {
        return this.processor.stringify(this._body);
      }
    },
    set: function(body) {
      if (this._data&&body) Errors.setBodyWithData(this);
      this._body = body;
      return this;
    },
    enumerable: true
  },

// - **processor**. The functions that will be used to convert to/from `data` and
//   `body` attributes. You can add processors. The two that are built-in are for
//   `text/plain`, which is basically an identity transformation and
//   `application/json` and other JSON-based media types (including custom media
//   types with `+json`). You can add your own processors. See below.
  processor: {
    get: function() {
      var processor = Content.processors[this.type];
      if (processor) {
        return processor;
      } else {
        // Return the first processor that matches any part of the
        // content type. ex: application/vnd.foobar.baz+json will match json.
        var main = this.type.split(";")[0];
        var parts = main.split(/\+|\//);
        for (var i=0, l=parts.length; i < l; i++) {
          processor = Content.processors[parts[i]]
        }
        return processor || {parser:identity,stringify:toString};
      }
    },
    enumerable: true
  },

// - **length**. Typically accessed as `content.length`, returns the length in
//   bytes of the raw content entity.
  length: {
    get: function() {
      if (typeof Buffer !== 'undefined') {
        return Buffer.byteLength(this.body);
      }
      return this.body.length;
    }
  }
});

Content.processors = {};

// The `registerProcessor` function allows you to add your own processors to
// convert content entities. Each processor consists of a Javascript object with
// two properties:
// - **parser**. The function used to parse a raw content entity and convert it
//   into a Javascript data type.
// - **stringify**. The function used to convert a Javascript data type into a
//   raw content entity.
Content.registerProcessor = function(types,processor) {
  
// You can pass an array of types that will trigger this processor, or just one.
// We determine the array via duck-typing here.
  if (types.forEach) {
    types.forEach(function(type) {
      Content.processors[type] = processor;
    });
  } else {
    // If you didn't pass an array, we just use what you pass in.
    Content.processors[types] = processor;
  }
};

// Register the identity processor, which is used for text-based media types.
var identity = function(x) { return x; }
  , toString = function(x) { return x.toString(); }
Content.registerProcessor(
  ["text/html","text/plain","text"],
  { parser: identity, stringify: toString });

// Register the JSON processor, which is used for JSON-based media types.
Content.registerProcessor(
  ["application/json; charset=utf-8","application/json","json"],
  {
    parser: function(string) {
      return JSON.parse(string);
    },
    stringify: function(data) {
      return JSON.stringify(data); }});

// Error functions are defined separately here in an attempt to make the code
// easier to read.
var Errors = {
  setDataWithBody: function(object) {
    throw new Error("Attempt to set data attribute of a content object " +
        "when the body attributes was already set.");
  },
  setBodyWithData: function(object) {
    throw new Error("Attempt to set body attribute of a content object " +
        "when the data attributes was already set.");
  }
}
module.exports = Content;

});

require.define("/shred/mixins/headers.js", function (require, module, exports, __dirname, __filename) {
    // The header mixins allow you to add HTTP header support to any object. This
// might seem pointless: why not simply use a hash? The main reason is that, per
// the [HTTP spec](http://www.w3.org/Protocols/rfc2616/rfc2616-sec4.html#sec4.2),
// headers are case-insensitive. So, for example, `content-type` is the same as
// `CONTENT-TYPE` which is the same as `Content-Type`. Since there is no way to
// overload the index operator in Javascript, using a hash to represent the
// headers means it's possible to have two conflicting values for a single
// header.
// 
// The solution to this is to provide explicit methods to set or get headers.
// This also has the benefit of allowing us to introduce additional variations,
// including snake case, which we automatically convert to what Matthew King has
// dubbed "corset case" - the hyphen-separated names with initial caps:
// `Content-Type`. We use corset-case just in case we're dealing with servers
// that haven't properly implemented the spec.

// Convert headers to corset-case. **Example:** `CONTENT-TYPE` will be converted
// to `Content-Type`.

var corsetCase = function(string) {
  return string;//.toLowerCase()
      //.replace("_","-")
      // .replace(/(^|-)(\w)/g, 
          // function(s) { return s.toUpperCase(); });
};

// We suspect that `initializeHeaders` was once more complicated ...
var initializeHeaders = function(object) {
  return {};
};

// Access the `_headers` property using lazy initialization. **Warning:** If you
// mix this into an object that is using the `_headers` property already, you're
// going to have trouble.
var $H = function(object) {
  return object._headers||(object._headers=initializeHeaders(object));
};

// Hide the implementations as private functions, separate from how we expose them.

// The "real" `getHeader` function: get the header after normalizing the name.
var getHeader = function(object,name) {
  return $H(object)[corsetCase(name)];
};

// The "real" `getHeader` function: get one or more headers, or all of them
// if you don't ask for any specifics. 
var getHeaders = function(object,names) {
  var keys = (names && names.length>0) ? names : Object.keys($H(object));
  var hash = keys.reduce(function(hash,key) {
    hash[key] = getHeader(object,key);
    return hash;
  },{});
  // Freeze the resulting hash so you don't mistakenly think you're modifying
  // the real headers.
  Object.freeze(hash);
  return hash;
};

// The "real" `setHeader` function: set a header, after normalizing the name.
var setHeader = function(object,name,value) {
  $H(object)[corsetCase(name)] = value;
  return object;
};

// The "real" `setHeaders` function: set multiple headers based on a hash.
var setHeaders = function(object,hash) {
  for( var key in hash ) { setHeader(object,key,hash[key]); };
  return this;
};

// Here's where we actually bind the functionality to an object. These mixins work by
// exposing mixin functions. Each function mixes in a specific batch of features.
module.exports = {
  
  // Add getters.
  getters: function(constructor) {
    constructor.prototype.getHeader = function(name) { return getHeader(this,name); };
    constructor.prototype.getHeaders = function() { return getHeaders(this,arguments); };
  },
  // Add setters but as "private" methods.
  privateSetters: function(constructor) {
    constructor.prototype._setHeader = function(key,value) { return setHeader(this,key,value); };
    constructor.prototype._setHeaders = function(hash) { return setHeaders(this,hash); };
  },
  // Add setters.
  setters: function(constructor) {
    constructor.prototype.setHeader = function(key,value) { return setHeader(this,key,value); };
    constructor.prototype.setHeaders = function(hash) { return setHeaders(this,hash); };
  },
  // Add both getters and setters.
  gettersAndSetters: function(constructor) {
    constructor.prototype.getHeader = function(name) { return getHeader(this,name); };
    constructor.prototype.getHeaders = function() { return getHeaders(this,arguments); };
    constructor.prototype.setHeader = function(key,value) { return setHeader(this,key,value); };
    constructor.prototype.setHeaders = function(hash) { return setHeaders(this,hash); };
  },
};

});

require.define("/node_modules/iconv-lite/package.json", function (require, module, exports, __dirname, __filename) {
    module.exports = {}
});

require.define("/node_modules/iconv-lite/index.js", function (require, module, exports, __dirname, __filename) {
    // Module exports
var iconv = module.exports = {
    toEncoding: function(str, encoding) {
        return iconv.getCodec(encoding).toEncoding(str);
    },
    fromEncoding: function(buf, encoding) {
        return iconv.getCodec(encoding).fromEncoding(buf);
    },
    
    defaultCharUnicode: '�',
    defaultCharSingleByte: '?',
    
    // Get correct codec for given encoding.
    getCodec: function(encoding) {
        var enc = encoding || "utf8";
        var codecOptions = undefined;
        while (1) {
            if (getType(enc) === "String")
                enc = enc.replace(/[- ]/g, "").toLowerCase();
            var codec = iconv.encodings[enc];
            var type = getType(codec);
            if (type === "String") {
                // Link to other encoding.
                codecOptions = {originalEncoding: enc};
                enc = codec;
            }
            else if (type === "Object" && codec.type != undefined) {
                // Options for other encoding.
                codecOptions = codec;
                enc = codec.type;
            } 
            else if (type === "Function")
                // Codec itself.
                return codec(codecOptions);
            else
                throw new Error("Encoding not recognized: '" + encoding + "' (searched as: '"+enc+"')");
        }
    },
    
    // Define basic encodings
    encodings: {
        internal: function(options) {
            return {
                toEncoding: function(str) {
                    return new Buffer(ensureString(str), options.originalEncoding);
                },
                fromEncoding: function(buf) {
                    return ensureBuffer(buf).toString(options.originalEncoding);
                }
            };
        },
        utf8: "internal",
        ucs2: "internal",
        binary: "internal",
        ascii: "internal",
        base64: "internal",
        
        // Codepage single-byte encodings.
        singlebyte: function(options) {
            // Prepare chars if needed
            if (!options.chars || (options.chars.length !== 128 && options.chars.length !== 256))
                throw new Error("Encoding '"+options.type+"' has incorrect 'chars' (must be of len 128 or 256)");
            
            if (options.chars.length === 128)
                options.chars = asciiString + options.chars;
            
            if (!options.charsBuf) {
                options.charsBuf = new Buffer(options.chars, 'ucs2');
            }
            
            if (!options.revCharsBuf) {
                options.revCharsBuf = new Buffer(65536);
                var defChar = iconv.defaultCharSingleByte.charCodeAt(0);
                for (var i = 0; i < options.revCharsBuf.length; i++)
                    options.revCharsBuf[i] = defChar;
                for (var i = 0; i < options.chars.length; i++)
                    options.revCharsBuf[options.chars.charCodeAt(i)] = i;
            }
            
            return {
                toEncoding: function(str) {
                    str = ensureString(str);
                    
                    var buf = new Buffer(str.length);
                    var revCharsBuf = options.revCharsBuf;
                    for (var i = 0; i < str.length; i++)
                        buf[i] = revCharsBuf[str.charCodeAt(i)];
                    
                    return buf;
                },
                fromEncoding: function(buf) {
                    buf = ensureBuffer(buf);
                    
                    // Strings are immutable in JS -> we use ucs2 buffer to speed up computations.
                    var charsBuf = options.charsBuf;
                    var newBuf = new Buffer(buf.length*2);
                    var idx1 = 0, idx2 = 0;
                    for (var i = 0, _len = buf.length; i < _len; i++) {
                        idx1 = buf[i]*2; idx2 = i*2;
                        newBuf[idx2] = charsBuf[idx1];
                        newBuf[idx2+1] = charsBuf[idx1+1];
                    }
                    return newBuf.toString('ucs2');
                }
            };
        },

        // Codepage double-byte encodings.
        table: function(options) {
            var table = options.table, key, revCharsTable = options.revCharsTable;
            if (!table) {
                throw new Error("Encoding '" + options.type +"' has incorect 'table' option");
            }
            if(!revCharsTable) {
                revCharsTable = options.revCharsTable = {};
                for (key in table) {
                    revCharsTable[table[key]] = parseInt(key);
                }
            }
            
            return {
                toEncoding: function(str) {
                    str = ensureString(str);
                    var strLen = str.length;
                    var bufLen = strLen;
                    for (var i = 0; i < strLen; i++)
                        if (str.charCodeAt(i) >> 7)
                            bufLen++;

                    var newBuf = new Buffer(bufLen), gbkcode, unicode, 
                        defaultChar = revCharsTable[iconv.defaultCharUnicode.charCodeAt(0)];

                    for (var i = 0, j = 0; i < strLen; i++) {
                        unicode = str.charCodeAt(i);
                        if (unicode >> 7) {
                            gbkcode = revCharsTable[unicode] || defaultChar;
                            newBuf[j++] = gbkcode >> 8; //high byte;
                            newBuf[j++] = gbkcode & 0xFF; //low byte
                        } else {//ascii
                            newBuf[j++] = unicode;
                        }
                    }
                    return newBuf;
                },
                fromEncoding: function(buf) {
                    buf = ensureBuffer(buf);
                    var bufLen = buf.length, strLen = 0;
                    for (var i = 0; i < bufLen; i++) {
                        strLen++;
                        if (buf[i] & 0x80) //the high bit is 1, so this byte is gbkcode's high byte.skip next byte
                            i++;
                    }
                    var newBuf = new Buffer(strLen*2), unicode, gbkcode,
                        defaultChar = iconv.defaultCharUnicode.charCodeAt(0);
                    
                    for (var i = 0, j = 0; i < bufLen; i++, j+=2) {
                        gbkcode = buf[i];
                        if (gbkcode & 0x80) {
                            gbkcode = (gbkcode << 8) + buf[++i];
                            unicode = table[gbkcode] || defaultChar;
                        } else {
                            unicode = gbkcode;
                        }
                        newBuf[j] = unicode & 0xFF; //low byte
                        newBuf[j+1] = unicode >> 8; //high byte
                    }
                    return newBuf.toString('ucs2');
                }
            }
        }
    }
};

// Add aliases to convert functions
iconv.encode = iconv.toEncoding;
iconv.decode = iconv.fromEncoding;

// Load other encodings from files in /encodings dir.
var encodingsDir = __dirname+"/encodings/",
    fs = require('fs');
fs.readdirSync(encodingsDir).forEach(function(file) {
    if(fs.statSync(encodingsDir + file).isDirectory()) return;
    var encodings = require(encodingsDir + file)
    for (var key in encodings)
        iconv.encodings[key] = encodings[key]
});

// Utilities
var asciiString = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'+
              ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f';

var ensureBuffer = function(buf) {
    buf = buf || new Buffer(0);
    return (buf instanceof Buffer) ? buf : new Buffer(buf.toString(), "utf8");
}

var ensureString = function(str) {
    str = str || "";
    return (str instanceof String) ? str : str.toString((str instanceof Buffer) ? 'utf8' : undefined);
}

var getType = function(obj) {
    return Object.prototype.toString.call(obj).slice(8, -1);
}


});

require.define("/node_modules/http-browserify/package.json", function (require, module, exports, __dirname, __filename) {
    module.exports = {"main":"index.js","browserify":"browser.js"}
});

require.define("/node_modules/http-browserify/browser.js", function (require, module, exports, __dirname, __filename) {
    var http = module.exports;
var EventEmitter = require('events').EventEmitter;
var Request = require('./lib/request');

http.request = function (params, cb) {
    if (!params) params = {};
    if (!params.host) params.host = window.location.host.split(':')[0];
    if (!params.port) params.port = window.location.port;
    
    var req = new Request(new xhrHttp, params);
    if (cb) req.on('response', cb);
    return req;
};

http.get = function (params, cb) {
    params.method = 'GET';
    var req = http.request(params, cb);
    req.end();
    return req;
};

var xhrHttp = (function () {
    if (typeof window === 'undefined') {
        throw new Error('no window object present');
    }
    else if (window.XMLHttpRequest) {
        return window.XMLHttpRequest;
    }
    else if (window.ActiveXObject) {
        var axs = [
            'Msxml2.XMLHTTP.6.0',
            'Msxml2.XMLHTTP.3.0',
            'Microsoft.XMLHTTP'
        ];
        for (var i = 0; i < axs.length; i++) {
            try {
                var ax = new(window.ActiveXObject)(axs[i]);
                return function () {
                    if (ax) {
                        var ax_ = ax;
                        ax = null;
                        return ax_;
                    }
                    else {
                        return new(window.ActiveXObject)(axs[i]);
                    }
                };
            }
            catch (e) {}
        }
        throw new Error('ajax not supported in this browser')
    }
    else {
        throw new Error('ajax not supported in this browser');
    }
})();

http.STATUS_CODES = {
    100 : 'Continue',
    101 : 'Switching Protocols',
    102 : 'Processing', // RFC 2518, obsoleted by RFC 4918
    200 : 'OK',
    201 : 'Created',
    202 : 'Accepted',
    203 : 'Non-Authoritative Information',
    204 : 'No Content',
    205 : 'Reset Content',
    206 : 'Partial Content',
    207 : 'Multi-Status', // RFC 4918
    300 : 'Multiple Choices',
    301 : 'Moved Permanently',
    302 : 'Moved Temporarily',
    303 : 'See Other',
    304 : 'Not Modified',
    305 : 'Use Proxy',
    307 : 'Temporary Redirect',
    400 : 'Bad Request',
    401 : 'Unauthorized',
    402 : 'Payment Required',
    403 : 'Forbidden',
    404 : 'Not Found',
    405 : 'Method Not Allowed',
    406 : 'Not Acceptable',
    407 : 'Proxy Authentication Required',
    408 : 'Request Time-out',
    409 : 'Conflict',
    410 : 'Gone',
    411 : 'Length Required',
    412 : 'Precondition Failed',
    413 : 'Request Entity Too Large',
    414 : 'Request-URI Too Large',
    415 : 'Unsupported Media Type',
    416 : 'Requested Range Not Satisfiable',
    417 : 'Expectation Failed',
    418 : 'I\'m a teapot', // RFC 2324
    422 : 'Unprocessable Entity', // RFC 4918
    423 : 'Locked', // RFC 4918
    424 : 'Failed Dependency', // RFC 4918
    425 : 'Unordered Collection', // RFC 4918
    426 : 'Upgrade Required', // RFC 2817
    500 : 'Internal Server Error',
    501 : 'Not Implemented',
    502 : 'Bad Gateway',
    503 : 'Service Unavailable',
    504 : 'Gateway Time-out',
    505 : 'HTTP Version not supported',
    506 : 'Variant Also Negotiates', // RFC 2295
    507 : 'Insufficient Storage', // RFC 4918
    509 : 'Bandwidth Limit Exceeded',
    510 : 'Not Extended' // RFC 2774
};

});

require.define("/node_modules/http-browserify/lib/request.js", function (require, module, exports, __dirname, __filename) {
    var EventEmitter = require('events').EventEmitter;
var Response = require('./response');
var isSafeHeader = require('./isSafeHeader');

var Request = module.exports = function (xhr, params) {
    var self = this;
    self.xhr = xhr;
    self.body = '';
    
    var uri = params.host + ':' + params.port + (params.path || '/');
    
    xhr.open(
        params.method || 'GET',
        (params.scheme || 'http') + '://' + uri,
        true
    );
    
    if (params.headers) {
        Object.keys(params.headers).forEach(function (key) {
            if (!isSafeHeader(key)) return;
            var value = params.headers[key];
            if (Array.isArray(value)) {
                value.forEach(function (v) {
                    xhr.setRequestHeader(key, v);
                });
            }
            else xhr.setRequestHeader(key, value)
        });
    }
    
    var res = new Response(xhr);
    res.on('ready', function () {
        self.emit('response', res);
    });
    
    xhr.onreadystatechange = function () {
        res.handle(xhr);
    };
};

Request.prototype = new EventEmitter;

Request.prototype.setHeader = function (key, value) {
    if ((Array.isArray && Array.isArray(value))
    || value instanceof Array) {
        for (var i = 0; i < value.length; i++) {
            this.xhr.setRequestHeader(key, value[i]);
        }
    }
    else {
        this.xhr.setRequestHeader(key, value);
    }
};

Request.prototype.write = function (s) {
    this.body += s;
};

Request.prototype.end = function (s) {
    if (s !== undefined) this.write(s);
    this.xhr.send(this.body);
};

});

require.define("/node_modules/http-browserify/lib/response.js", function (require, module, exports, __dirname, __filename) {
    var EventEmitter = require('events').EventEmitter;
var isSafeHeader = require('./isSafeHeader');

var Response = module.exports = function (xhr) {
    this.xhr = xhr;
    this.offset = 0;
};

Response.prototype = new EventEmitter;

var capable = {
    streaming : true,
    status2 : true
};

function parseHeaders (xhr) {
    var lines = xhr.getAllResponseHeaders().split(/\r?\n/);
    var headers = {};
    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        if (line === '') continue;
        
        var m = line.match(/^([^:]+):\s*(.*)/);
        if (m) {
            var key = m[1].toLowerCase(), value = m[2];
            
            if (headers[key] !== undefined) {
                if ((Array.isArray && Array.isArray(headers[key]))
                || headers[key] instanceof Array) {
                    headers[key].push(value);
                }
                else {
                    headers[key] = [ headers[key], value ];
                }
            }
            else {
                headers[key] = value;
            }
        }
        else {
            headers[line] = true;
        }
    }
    return headers;
}

Response.prototype.getHeader = function (key) {
    var header = this.headers ? this.headers[key.toLowerCase()] : null;
    if (header) return header;

    // Work around Mozilla bug #608735 [https://bugzil.la/608735], which causes
    // getAllResponseHeaders() to return {} if the response is a CORS request.
    // xhr.getHeader still works correctly.
    if (isSafeHeader(key)) {
      return this.xhr.getResponseHeader(key);
    }
    return null;
};

Response.prototype.handle = function () {
    var xhr = this.xhr;
    if (xhr.readyState === 2 && capable.status2) {
        try {
            this.statusCode = xhr.status;
            this.headers = parseHeaders(xhr);
        }
        catch (err) {
            capable.status2 = false;
        }
        
        if (capable.status2) {
            this.emit('ready');
        }
    }
    else if (capable.streaming && xhr.readyState === 3) {
        try {
            if (!this.statusCode) {
                this.statusCode = xhr.status;
                this.headers = parseHeaders(xhr);
                this.emit('ready');
            }
        }
        catch (err) {}
        
        try {
            this.write();
        }
        catch (err) {
            capable.streaming = false;
        }
    }
    else if (xhr.readyState === 4) {
        if (!this.statusCode) {
            this.statusCode = xhr.status;
            this.emit('ready');
        }
        this.write();
        
        if (xhr.error) {
            this.emit('error', xhr.responseText);
        }
        else this.emit('end');
    }
};

Response.prototype.write = function () {
    var xhr = this.xhr;
    if (xhr.responseText.length > this.offset) {
        this.emit('data', xhr.responseText.slice(this.offset));
        this.offset = xhr.responseText.length;
    }
};

});

require.define("/node_modules/http-browserify/lib/isSafeHeader.js", function (require, module, exports, __dirname, __filename) {
    // Taken from http://dxr.mozilla.org/mozilla/mozilla-central/content/base/src/nsXMLHttpRequest.cpp.html
var unsafeHeaders = [
    "accept-charset",
    "accept-encoding",
    "access-control-request-headers",
    "access-control-request-method",
    "connection",
    "content-length",
    "cookie",
    "cookie2",
    "content-transfer-encoding",
    "date",
    "expect",
    "host",
    "keep-alive",
    "origin",
    "referer",
    "set-cookie",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "user-agent",
    "via"
];

module.exports = function (headerName) {
    if (!headerName) return false;
    return (unsafeHeaders.indexOf(headerName.toLowerCase()) === -1)
};

});

require.alias("http-browserify", "/node_modules/http");

require.alias("http-browserify", "/node_modules/https");