'use strict'

var http = require('http')
  , https = require('https')
  , url = require('url')
  , util = require('util')
  , stream = require('stream')
  , qs = require('qs')
  , querystring = require('querystring')
  , zlib = require('zlib')
  , helpers = require('./lib/helpers')
  , bl = require('bl')
  , hawk = require('hawk')
  , aws = require('aws-sign2')
  , httpSignature = require('http-signature')
  , uuid = require('node-uuid')
  , mime = require('mime-types')
  , tunnel = require('tunnel-agent')
  , stringstream = require('stringstream')
  , caseless = require('caseless')
  , ForeverAgent = require('forever-agent')
  , FormData = require('form-data')
  , cookies = require('./lib/cookies')
  , copy = require('./lib/copy')
  , net = require('net')
  , CombinedStream = require('combined-stream')
  , isstream = require('isstream')
  , getProxyFromURI = require('./lib/getProxyFromURI')
  , Auth = require('./lib/auth').Auth
  , oauth = require('./lib/oauth')

var safeStringify = helpers.safeStringify
  , md5 = helpers.md5
  , isReadStream = helpers.isReadStream
  , toBase64 = helpers.toBase64
  , defer = helpers.defer
  , globalCookieJar = cookies.jar()


var globalPool = {}
  , isUrl = /^https?:/

var defaultProxyHeaderWhiteList = [
  'accept',
  'accept-charset',
  'accept-encoding',
  'accept-language',
  'accept-ranges',
  'cache-control',
  'content-encoding',
  'content-language',
  'content-length',
  'content-location',
  'content-md5',
  'content-range',
  'content-type',
  'connection',
  'date',
  'expect',
  'max-forwards',
  'pragma',
  'referer',
  'te',
  'transfer-encoding',
  'user-agent',
  'via'
]

var defaultProxyHeaderExclusiveList = [
  'proxy-authorization'
]

function filterForNonReserved(reserved, options) {
  // Filter out properties that are not reserved.
  // Reserved values are passed in at call site.

  var object = {}
  for (var i in options) {
    var notReserved = (reserved.indexOf(i) === -1)
    if (notReserved) {
      object[i] = options[i]
    }
  }
  return object
}

function filterOutReservedFunctions(reserved, options) {
  // Filter out properties that are functions and are reserved.
  // Reserved values are passed in at call site.

  var object = {}
  for (var i in options) {
    var isReserved = !(reserved.indexOf(i) === -1)
    var isFunction = (typeof options[i] === 'function')
    if (!(isReserved && isFunction)) {
      object[i] = options[i]
    }
  }
  return object

}

function constructProxyHost(uriObject) {
  var port = uriObject.portA
    , protocol = uriObject.protocol
    , proxyHost = uriObject.hostname + ':'

  if (port) {
    proxyHost += port
  } else if (protocol === 'https:') {
    proxyHost += '443'
  } else {
    proxyHost += '80'
  }

  return proxyHost
}

function constructProxyHeaderWhiteList(headers, proxyHeaderWhiteList) {
  var whiteList = proxyHeaderWhiteList
    .reduce(function (set, header) {
      set[header.toLowerCase()] = true
      return set
    }, {})

  return Object.keys(headers)
    .filter(function (header) {
      return whiteList[header.toLowerCase()]
    })
    .reduce(function (set, header) {
      set[header] = headers[header]
      return set
    }, {})
}

function getTunnelOption(self, options) {
  // Tunnel HTTPS by default, or if a previous request in the redirect chain
  // was tunneled.  Allow the user to override this setting.

  // If self.tunnel is already set (because this is a redirect), use the
  // existing value.
  if (typeof self.tunnel !== 'undefined') {
    return self.tunnel
  }

  // If options.tunnel is set (the user specified a value), use it.
  if (typeof options.tunnel !== 'undefined') {
    return options.tunnel
  }

  // If the destination is HTTPS, tunnel.
  if (self.uri.protocol === 'https:') {
    return true
  }

  // Otherwise, leave tunnel unset, because if a later request in the redirect
  // chain is HTTPS then that request (and any subsequent ones) should be
  // tunneled.
  return undefined
}

function constructTunnelOptions(request) {
  var proxy = request.proxy

  var tunnelOptions = {
    proxy : {
      host      : proxy.hostname,
      port      : +proxy.port,
      proxyAuth : proxy.auth,
      headers   : request.proxyHeaders
    },
    headers            : request.headers,
    ca                 : request.ca,
    cert               : request.cert,
    key                : request.key,
    passphrase         : request.passphrase,
    pfx                : request.pfx,
    ciphers            : request.ciphers,
    rejectUnauthorized : request.rejectUnauthorized,
    secureOptions      : request.secureOptions,
    secureProtocol     : request.secureProtocol
  }

  return tunnelOptions
}

function constructTunnelFnName(uri, proxy) {
  var uriProtocol = (uri.protocol === 'https:' ? 'https' : 'http')
  var proxyProtocol = (proxy.protocol === 'https:' ? 'Https' : 'Http')
  return [uriProtocol, proxyProtocol].join('Over')
}

function getTunnelFn(request) {
  var uri = request.uri
  var proxy = request.proxy
  var tunnelFnName = constructTunnelFnName(uri, proxy)
  return tunnel[tunnelFnName]
}

// Function for properly handling a connection error
function connectionErrorHandler(error) {
  var socket = this
  if (socket.res) {
    if (socket.res.request) {
      socket.res.request.emit('error', error)
    } else {
      socket.res.emit('error', error)
    }
  } else {
    socket._httpMessage.emit('error', error)
  }
}

// Return a simpler request object to allow serialization
function requestToJSON() {
  var self = this
  return {
    uri: self.uri,
    method: self.method,
    headers: self.headers
  }
}

// Return a simpler response object to allow serialization
function responseToJSON() {
  var self = this
  return {
    statusCode: self.statusCode,
    body: self.body,
    headers: self.headers,
    request: requestToJSON.call(self.request)
  }
}

// encode rfc3986 characters
function rfc3986 (str) {
  return str.replace(/[!'()*]/g, function(c) {
    return '%' + c.charCodeAt(0).toString(16).toUpperCase()
  })
}

function Request (options) {
  // if given the method property in options, set property explicitMethod to true

  // extend the Request instance with any non-reserved properties
  // remove any reserved functions from the options object
  // set Request instance to be readable and writable
  // call init

  var self = this
  stream.Stream.call(self)
  var reserved = Object.keys(Request.prototype)
  var nonReserved = filterForNonReserved(reserved, options)

  stream.Stream.call(self)
  util._extend(self, nonReserved)
  options = filterOutReservedFunctions(reserved, options)

  self.readable = true
  self.writable = true
  if (options.method) {
    self.explicitMethod = true
  }
  self.init(options)
}

util.inherits(Request, stream.Stream)

// Debugging
Request.debug = process.env.NODE_DEBUG && /\brequest\b/.test(process.env.NODE_DEBUG)
function debug() {
  if (Request.debug) {
    console.error('REQUEST %s', util.format.apply(util, arguments))
  }
}

Request.prototype.setupTunnel = function () {
  var self = this

  if (typeof self.proxy === 'string') {
    self.proxy = url.parse(self.proxy)
  }
  
  if (!self.proxy || !self.tunnel) {
    return false
  }

  // Setup Proxy Header Exclusive List and White List
  self.proxyHeaderExclusiveList = self.proxyHeaderExclusiveList || []
  self.proxyHeaderWhiteList = self.proxyHeaderWhiteList || defaultProxyHeaderWhiteList
  var proxyHeaderExclusiveList = self.proxyHeaderExclusiveList.concat(defaultProxyHeaderExclusiveList)
  var proxyHeaderWhiteList = self.proxyHeaderWhiteList.concat(proxyHeaderExclusiveList)

  // Setup Proxy Headers and Proxy Headers Host
  // Only send the Proxy White Listed Header names
  self.proxyHeaders = constructProxyHeaderWhiteList(self.headers, proxyHeaderWhiteList)
  self.proxyHeaders.host = constructProxyHost(self.uri)
  proxyHeaderExclusiveList.forEach(self.removeHeader, self)
 
  // Set Agent from Tunnel Data
  var tunnelFn = getTunnelFn(self)
  var tunnelOptions = constructTunnelOptions(self)
  self.agent = tunnelFn(tunnelOptions)

  return true
}

Request.prototype.init = function (options) {
  // init() contains all the code to setup the request object.
  // the actual outgoing request is not started until start() is called
  // this function is called from both the constructor and on redirect.
  var self = this
  if (!options) {
    options = {}
  }
  self.headers = self.headers ? copy(self.headers) : {}

  caseless.httpify(self, self.headers)

  if (!self.method) {
    self.method = options.method || 'GET'
  }
  self.localAddress = options.localAddress

  if (!self.qsLib) {
    self.qsLib = (options.useQuerystring ? querystring : qs)
  }

  debug(options)
  if (!self.pool && self.pool !== false) {
    self.pool = globalPool
  }
  self.dests = self.dests || []
  self.__isRequestRequest = true

  // Protect against double callback
  if (!self._callback && self.callback) {
    self._callback = self.callback
    self.callback = function () {
      if (self._callbackCalled) {
        return // Print a warning maybe?
      }
      self._callbackCalled = true
      self._callback.apply(self, arguments)
    }
    self.on('error', self.callback.bind())
    self.on('complete', self.callback.bind(self, null))
  }

  // People use this property instead all the time, so support it
  if (!self.uri && self.url) {
    self.uri = self.url
    delete self.url
  }

  // A URI is needed by this point, throw if we haven't been able to get one
  if (!self.uri) {
    return self.emit('error', new Error('options.uri is a required argument'))
  }

  // If a string URI/URL was given, parse it into a URL object
  if(typeof self.uri === 'string') {
    self.uri = url.parse(self.uri)
  }

  // DEPRECATED: Warning for users of the old Unix Sockets URL Scheme
  if (self.uri.protocol === 'unix:') {
    return self.emit('error', new Error('`unix://` URL scheme is no longer supported. Please use the format `http://unix:SOCKET:PATH`'))
  }

  // Support Unix Sockets
  if(self.uri.host === 'unix') {
    // Get the socket & request paths from the URL
    var unixParts = self.uri.path.split(':')
      , host = unixParts[0]
      , path = unixParts[1]
    // Apply unix properties to request
    self.socketPath = host
    self.uri.pathname = path
    self.uri.path = path
    self.uri.host = host
    self.uri.hostname = host
    self.uri.isUnix = true
  }

  if (self.strictSSL === false) {
    self.rejectUnauthorized = false
  }

  if(!self.hasOwnProperty('proxy')) {
    self.proxy = getProxyFromURI(self.uri)
  }

  self.tunnel = getTunnelOption(self, options)
  if (self.proxy) {
    self.setupTunnel()
  }

  if (!self.uri.pathname) {self.uri.pathname = '/'}

  if (!(self.uri.host || (self.uri.hostname && self.uri.port)) && !self.uri.isUnix) {
    // Invalid URI: it may generate lot of bad errors, like 'TypeError: Cannot call method `indexOf` of undefined' in CookieJar
    // Detect and reject it as soon as possible
    var faultyUri = url.format(self.uri)
    var message = 'Invalid URI "' + faultyUri + '"'
    if (Object.keys(options).length === 0) {
      // No option ? This can be the sign of a redirect
      // As this is a case where the user cannot do anything (they didn't call request directly with this URL)
      // they should be warned that it can be caused by a redirection (can save some hair)
      message += '. This can be caused by a crappy redirection.'
    }
    // This error was fatal
    return self.emit('error', new Error(message))
  }

  self._redirectsFollowed = self._redirectsFollowed || 0
  self.maxRedirects = (self.maxRedirects !== undefined) ? self.maxRedirects : 10
  self.allowRedirect = (typeof self.followRedirect === 'function') ? self.followRedirect : function(response) {
    return true
  }
  self.followRedirects = (self.followRedirect !== undefined) ? !!self.followRedirect : true
  self.followAllRedirects = (self.followAllRedirects !== undefined) ? self.followAllRedirects : false
  if (self.followRedirects || self.followAllRedirects) {
    self.redirects = self.redirects || []
  }

  self.setHost = false
  if (!self.hasHeader('host')) {
    var hostHeaderName = self.originalHostHeaderName || 'host'
    self.setHeader(hostHeaderName, self.uri.hostname)
    if (self.uri.port) {
      if ( !(self.uri.port === 80 && self.uri.protocol === 'http:') &&
           !(self.uri.port === 443 && self.uri.protocol === 'https:') ) {
        self.setHeader(hostHeaderName, self.getHeader('host') + (':' + self.uri.port) )
      }
    }
    self.setHost = true
  }

  self.jar(self._jar || options.jar)

  if (!self.uri.port) {
    if (self.uri.protocol === 'http:') {self.uri.port = 80}
    else if (self.uri.protocol === 'https:') {self.uri.port = 443}
  }

  if (self.proxy && !self.tunnel) {
    self.port = self.proxy.port
    self.host = self.proxy.hostname
  } else {
    self.port = self.uri.port
    self.host = self.uri.hostname
  }

  if (options.form) {
    self.form(options.form)
  }

  if (options.formData) {
    var formData = options.formData
    var requestForm = self.form()
    var appendFormValue = function (key, value) {
      if (value.hasOwnProperty('value') && value.hasOwnProperty('options')) {
        requestForm.append(key, value.value, value.options)
      } else {
        requestForm.append(key, value)
      }
    }
    for (var formKey in formData) {
      if (formData.hasOwnProperty(formKey)) {
        var formValue = formData[formKey]
        if (formValue instanceof Array) {
          for (var j = 0; j < formValue.length; j++) {
            appendFormValue(formKey, formValue[j])
          }
        } else {
          appendFormValue(formKey, formValue)
        }
      }
    }
  }

  if (options.qs) {
    self.qs(options.qs)
  }

  if (self.uri.path) {
    self.path = self.uri.path
  } else {
    self.path = self.uri.pathname + (self.uri.search || '')
  }

  if (self.path.length === 0) {
    self.path = '/'
  }

  // Auth must happen last in case signing is dependent on other headers
  self._auth = new Auth()

  if (options.oauth) {
    self.oauth(options.oauth)
  }

  if (options.aws) {
    self.aws(options.aws)
  }

  if (options.hawk) {
    self.hawk(options.hawk)
  }

  if (options.httpSignature) {
    self.httpSignature(options.httpSignature)
  }

  if (options.auth) {
    if (Object.prototype.hasOwnProperty.call(options.auth, 'username')) {
      options.auth.user = options.auth.username
    }
    if (Object.prototype.hasOwnProperty.call(options.auth, 'password')) {
      options.auth.pass = options.auth.password
    }

    self.auth(
      options.auth.user,
      options.auth.pass,
      options.auth.sendImmediately,
      options.auth.bearer
    )
  }

  if (self.gzip && !self.hasHeader('accept-encoding')) {
    self.setHeader('accept-encoding', 'gzip')
  }

  if (self.uri.auth && !self.hasHeader('authorization')) {
    var uriAuthPieces = self.uri.auth.split(':').map(function(item){ return querystring.unescape(item) })
    self.auth(uriAuthPieces[0], uriAuthPieces.slice(1).join(':'), true)
  }

  if (!self.tunnel && self.proxy && self.proxy.auth && !self.hasHeader('proxy-authorization')) {
    var proxyAuthPieces = self.proxy.auth.split(':').map(function(item){
      return querystring.unescape(item)
    })
    var authHeader = 'Basic ' + toBase64(proxyAuthPieces.join(':'))
    self.setHeader('proxy-authorization', authHeader)
  }

  if (self.proxy && !self.tunnel) {
    self.path = (self.uri.protocol + '//' + self.uri.host + self.path)
  }

  if (options.json) {
    self.json(options.json)
  }
  if (options.multipart) {
    self.boundary = uuid()
    self.multipart(options.multipart)
  }

  if (self.body) {
    var length = 0
    if (!Buffer.isBuffer(self.body)) {
      if (Array.isArray(self.body)) {
        for (var i = 0; i < self.body.length; i++) {
          length += self.body[i].length
        }
      } else {
        self.body = new Buffer(self.body)
        length = self.body.length
      }
    } else {
      length = self.body.length
    }
    if (length) {
      if (!self.hasHeader('content-length')) {
        self.setHeader('content-length', length)
      }
    } else {
      throw new Error('Argument error, options.body.')
    }
  }

  var protocol = self.proxy && !self.tunnel ? self.proxy.protocol : self.uri.protocol
    , defaultModules = {'http:':http, 'https:':https}
    , httpModules = self.httpModules || {}

  self.httpModule = httpModules[protocol] || defaultModules[protocol]

  if (!self.httpModule) {
    return self.emit('error', new Error('Invalid protocol: ' + protocol))
  }

  if (options.ca) {
    self.ca = options.ca
  }

  if (!self.agent) {
    if (options.agentOptions) {
      self.agentOptions = options.agentOptions
    }

    if (options.agentClass) {
      self.agentClass = options.agentClass
    } else if (options.forever) {
      self.agentClass = protocol === 'http:' ? ForeverAgent : ForeverAgent.SSL
    } else {
      self.agentClass = self.httpModule.Agent
    }
  }

  if (self.pool === false) {
    self.agent = false
  } else {
    self.agent = self.agent || self.getNewAgent()
  }

  self.on('pipe', function (src) {
    if (self.ntick && self._started) {
      throw new Error('You cannot pipe to this stream after the outbound request has started.')
    }
    self.src = src
    if (isReadStream(src)) {
      if (!self.hasHeader('content-type')) {
        self.setHeader('content-type', mime.lookup(src.path))
      }
    } else {
      if (src.headers) {
        for (var i in src.headers) {
          if (!self.hasHeader(i)) {
            self.setHeader(i, src.headers[i])
          }
        }
      }
      if (self._json && !self.hasHeader('content-type')) {
        self.setHeader('content-type', 'application/json')
      }
      if (src.method && !self.explicitMethod) {
        self.method = src.method
      }
    }

    // self.on('pipe', function () {
    //   console.error('You have already piped to this stream. Pipeing twice is likely to break the request.')
    // })
  })

  defer(function () {
    if (self._aborted) {
      return
    }

    var end = function () {
      if (self._form) {
        self._form.pipe(self)
      }
      if (self._multipart) {
        self._multipart.pipe(self)
      }
      if (self.body) {
        if (Array.isArray(self.body)) {
          self.body.forEach(function (part) {
            self.write(part)
          })
        } else {
          self.write(self.body)
        }
        self.end()
      } else if (self.requestBodyStream) {
        console.warn('options.requestBodyStream is deprecated, please pass the request object to stream.pipe.')
        self.requestBodyStream.pipe(self)
      } else if (!self.src) {
        if (self.method !== 'GET' && typeof self.method !== 'undefined') {
          self.setHeader('content-length', 0)
        }
        self.end()
      }
    }

    if (self._form && !self.hasHeader('content-length')) {
      // Before ending the request, we had to compute the length of the whole form, asyncly
      self.setHeader(self._form.getHeaders())
      self._form.getLength(function (err, length) {
        if (!err) {
          self.setHeader('content-length', length)
        }
        end()
      })
    } else {
      end()
    }

    self.ntick = true
  })

}

// Must call this when following a redirect from https to http or vice versa
// Attempts to keep everything as identical as possible, but update the
// httpModule, Tunneling agent, and/or Forever Agent in use.
Request.prototype._updateProtocol = function () {
  var self = this
  var protocol = self.uri.protocol

  if (protocol === 'https:' || self.tunnel) {
    // previously was doing http, now doing https
    // if it's https, then we might need to tunnel now.
    if (self.proxy) {
      if (self.setupTunnel()) {
        return
      }
    }

    self.httpModule = https
    switch (self.agentClass) {
      case ForeverAgent:
        self.agentClass = ForeverAgent.SSL
        break
      case http.Agent:
        self.agentClass = https.Agent
        break
      default:
        // nothing we can do.  Just hope for the best.
        return
    }

    // if there's an agent, we need to get a new one.
    if (self.agent) {
      self.agent = self.getNewAgent()
    }

  } else {
    // previously was doing https, now doing http
    self.httpModule = http
    switch (self.agentClass) {
      case ForeverAgent.SSL:
        self.agentClass = ForeverAgent
        break
      case https.Agent:
        self.agentClass = http.Agent
        break
      default:
        // nothing we can do.  just hope for the best
        return
    }

    // if there's an agent, then get a new one.
    if (self.agent) {
      self.agent = null
      self.agent = self.getNewAgent()
    }
  }
}

Request.prototype.getNewAgent = function () {
  var self = this
  var Agent = self.agentClass
  var options = {}
  if (self.agentOptions) {
    for (var i in self.agentOptions) {
      options[i] = self.agentOptions[i]
    }
  }
  if (self.ca) {
    options.ca = self.ca
  }
  if (self.ciphers) {
    options.ciphers = self.ciphers
  }
  if (self.secureProtocol) {
    options.secureProtocol = self.secureProtocol
  }
  if (self.secureOptions) {
    options.secureOptions = self.secureOptions
  }
  if (typeof self.rejectUnauthorized !== 'undefined') {
    options.rejectUnauthorized = self.rejectUnauthorized
  }

  if (self.cert && self.key) {
    options.key = self.key
    options.cert = self.cert
  }

  if (self.pfx) {
    options.pfx = self.pfx
  }

  if (self.passphrase) {
    options.passphrase = self.passphrase
  }

  var poolKey = ''

  // different types of agents are in different pools
  if (Agent !== self.httpModule.Agent) {
    poolKey += Agent.name
  }

  // ca option is only relevant if proxy or destination are https
  var proxy = self.proxy
  if (typeof proxy === 'string') {
    proxy = url.parse(proxy)
  }
  var isHttps = (proxy && proxy.protocol === 'https:') || this.uri.protocol === 'https:'

  if (isHttps) {
    if (options.ca) {
      if (poolKey) {
        poolKey += ':'
      }
      poolKey += options.ca
    }

    if (typeof options.rejectUnauthorized !== 'undefined') {
      if (poolKey) {
        poolKey += ':'
      }
      poolKey += options.rejectUnauthorized
    }

    if (options.cert) {
      if (poolKey) {
        poolKey += ':'
      }
      poolKey += options.cert.toString('ascii') + options.key.toString('ascii')
    }

    if (options.pfx) {
      if (poolKey) {
        poolKey += ':'
      }
      poolKey += options.pfx.toString('ascii')
    }

    if (options.ciphers) {
      if (poolKey) {
        poolKey += ':'
      }
      poolKey += options.ciphers
    }

    if (options.secureProtocol) {
      if (poolKey) {
        poolKey += ':'
      }
      poolKey += options.secureProtocol
    }

    if (options.secureOptions) {
      if (poolKey) {
        poolKey += ':'
      }
      poolKey += options.secureOptions
    }
  }

  if (self.pool === globalPool && !poolKey && Object.keys(options).length === 0 && self.httpModule.globalAgent) {
    // not doing anything special.  Use the globalAgent
    return self.httpModule.globalAgent
  }

  // we're using a stored agent.  Make sure it's protocol-specific
  poolKey = self.uri.protocol + poolKey

  // generate a new agent for this setting if none yet exists
  if (!self.pool[poolKey]) {
    self.pool[poolKey] = new Agent(options)
    // properly set maxSockets on new agents
    if (self.pool.maxSockets) {
      self.pool[poolKey].maxSockets = self.pool.maxSockets
    }
  }

  return self.pool[poolKey]
}

Request.prototype.start = function () {
  // start() is called once we are ready to send the outgoing HTTP request.
  // this is usually called on the first write(), end() or on nextTick()
  var self = this

  if (self._aborted) {
    return
  }

  self._started = true
  self.method = self.method || 'GET'
  self.href = self.uri.href

  if (self.src && self.src.stat && self.src.stat.size && !self.hasHeader('content-length')) {
    self.setHeader('content-length', self.src.stat.size)
  }
  if (self._aws) {
    self.aws(self._aws, true)
  }

  // We have a method named auth, which is completely different from the http.request
  // auth option.  If we don't remove it, we're gonna have a bad time.
  var reqOptions = copy(self)
  delete reqOptions.auth

  debug('make request', self.uri.href)
  self.req = self.httpModule.request(reqOptions)

  if (self.timeout && !self.timeoutTimer) {
    self.timeoutTimer = setTimeout(function () {
      self.abort()
      var e = new Error('ETIMEDOUT')
      e.code = 'ETIMEDOUT'
      self.emit('error', e)
    }, self.timeout)

    // Set additional timeout on socket - in case if remote
    // server freeze after sending headers
    if (self.req.setTimeout) { // only works on node 0.6+
      self.req.setTimeout(self.timeout, function () {
        if (self.req) {
          self.req.abort()
          var e = new Error('ESOCKETTIMEDOUT')
          e.code = 'ESOCKETTIMEDOUT'
          self.emit('error', e)
        }
      })
    }
  }

  self.req.on('response', self.onRequestResponse.bind(self))
  self.req.on('error', self.onRequestError.bind(self))
  self.req.on('drain', function() {
    self.emit('drain')
  })
  self.req.on('socket', function(socket) {
    self.emit('socket', socket)
  })

  self.on('end', function() {
    if ( self.req.connection ) {
      self.req.connection.removeListener('error', connectionErrorHandler)
    }
  })
  self.emit('request', self.req)
}

Request.prototype.onRequestError = function (error) {
  var self = this
  if (self._aborted) {
    return
  }
  if (self.req && self.req._reusedSocket && error.code === 'ECONNRESET'
      && self.agent.addRequestNoreuse) {
    self.agent = { addRequest: self.agent.addRequestNoreuse.bind(self.agent) }
    self.start()
    self.req.end()
    return
  }
  if (self.timeout && self.timeoutTimer) {
    clearTimeout(self.timeoutTimer)
    self.timeoutTimer = null
  }
  self.emit('error', error)
}

Request.prototype.onRequestResponse = function (response) {
  var self = this
  debug('onRequestResponse', self.uri.href, response.statusCode, response.headers)
  response.on('end', function() {
    debug('response end', self.uri.href, response.statusCode, response.headers)
  })

  // The check on response.connection is a workaround for browserify.
  if (response.connection && response.connection.listeners('error').indexOf(connectionErrorHandler) === -1) {
    response.connection.setMaxListeners(0)
    response.connection.once('error', connectionErrorHandler)
  }
  if (self._aborted) {
    debug('aborted', self.uri.href)
    response.resume()
    return
  }
  if (self._paused) {
    response.pause()
  } else if (response.resume) {
    // response.resume should be defined, but check anyway before calling. Workaround for browserify.
    response.resume()
  }

  self.response = response
  response.request = self
  response.toJSON = responseToJSON

  // XXX This is different on 0.10, because SSL is strict by default
  if (self.httpModule === https &&
      self.strictSSL && (!response.hasOwnProperty('client') ||
      !response.client.authorized)) {
    debug('strict ssl error', self.uri.href)
    var sslErr = response.hasOwnProperty('client') ? response.client.authorizationError : self.uri.href + ' does not support SSL'
    self.emit('error', new Error('SSL Error: ' + sslErr))
    return
  }

  // Save the original host before any redirect (if it changes, we need to
  // remove any authorization headers).  Also remember the case of the header
  // name because lots of broken servers expect Host instead of host and we
  // want the caller to be able to specify this.
  self.originalHost = self.getHeader('host')
  if (!self.originalHostHeaderName) {
    self.originalHostHeaderName = self.hasHeader('host')
  }
  if (self.setHost) {
    self.removeHeader('host')
  }
  if (self.timeout && self.timeoutTimer) {
    clearTimeout(self.timeoutTimer)
    self.timeoutTimer = null
  }

  var targetCookieJar = (self._jar && self._jar.setCookie) ? self._jar : globalCookieJar
  var addCookie = function (cookie) {
    //set the cookie if it's domain in the href's domain.
    try {
      targetCookieJar.setCookie(cookie, self.uri.href, {ignoreError: true})
    } catch (e) {
      self.emit('error', e)
    }
  }

  response.caseless = caseless(response.headers)

  if (response.caseless.has('set-cookie') && (!self._disableCookies)) {
    var headerName = response.caseless.has('set-cookie')
    if (Array.isArray(response.headers[headerName])) {
      response.headers[headerName].forEach(addCookie)
    } else {
      addCookie(response.headers[headerName])
    }
  }

  var redirectTo = null
  if (response.statusCode >= 300 && response.statusCode < 400 && response.caseless.has('location')) {
    var location = response.caseless.get('location')
    debug('redirect', location)

    if (self.followAllRedirects) {
      redirectTo = location
    } else if (self.followRedirects) {
      switch (self.method) {
        case 'PATCH':
        case 'PUT':
        case 'POST':
        case 'DELETE':
          // Do not follow redirects
          break
        default:
          redirectTo = location
          break
      }
    }
  } else if (response.statusCode === 401) {
    var authHeader = self._auth.response(self.method, self.uri.path, response.headers)
    if (authHeader) {
      self.setHeader('authorization', authHeader)
      redirectTo = self.uri
    }
  }

  if (redirectTo && self.allowRedirect.call(self, response)) {
    debug('redirect to', redirectTo)

    // ignore any potential response body.  it cannot possibly be useful
    // to us at this point.
    if (self._paused) {
      response.resume()
    }

    if (self._redirectsFollowed >= self.maxRedirects) {
      self.emit('error', new Error('Exceeded maxRedirects. Probably stuck in a redirect loop ' + self.uri.href))
      return
    }
    self._redirectsFollowed += 1

    if (!isUrl.test(redirectTo)) {
      redirectTo = url.resolve(self.uri.href, redirectTo)
    }

    var uriPrev = self.uri
    self.uri = url.parse(redirectTo)

    // handle the case where we change protocol from https to http or vice versa
    if (self.uri.protocol !== uriPrev.protocol) {
      self._updateProtocol()
    }

    self.redirects.push(
      { statusCode : response.statusCode
      , redirectUri: redirectTo
      }
    )
    if (self.followAllRedirects && response.statusCode !== 401 && response.statusCode !== 307) {
      self.method = 'GET'
    }
    // self.method = 'GET' // Force all redirects to use GET || commented out fixes #215
    delete self.src
    delete self.req
    delete self.agent
    delete self._started
    if (response.statusCode !== 401 && response.statusCode !== 307) {
      // Remove parameters from the previous response, unless this is the second request
      // for a server that requires digest authentication.
      delete self.body
      delete self._form
      if (self.headers) {
        self.removeHeader('host')
        self.removeHeader('content-type')
        self.removeHeader('content-length')
        if (self.uri.hostname !== self.originalHost.split(':')[0]) {
          // Remove authorization if changing hostnames (but not if just
          // changing ports or protocols).  This matches the behavior of curl:
          // https://github.com/bagder/curl/blob/6beb0eee/lib/http.c#L710
          self.removeHeader('authorization')
        }
      }
    }

    self.emit('redirect')

    self.init()
    return // Ignore the rest of the response
  } else {
    self._redirectsFollowed = self._redirectsFollowed || 0
    // Be a good stream and emit end when the response is finished.
    // Hack to emit end on close because of a core bug that never fires end
    response.on('close', function () {
      if (!self._ended) {
        self.response.emit('end')
      }
    })

    response.on('end', function () {
      self._ended = true
    })

    var dataStream
    if (self.gzip) {
      var contentEncoding = response.headers['content-encoding'] || 'identity'
      contentEncoding = contentEncoding.trim().toLowerCase()

      if (contentEncoding === 'gzip') {
        dataStream = zlib.createGunzip()
        response.pipe(dataStream)
      } else {
        // Since previous versions didn't check for Content-Encoding header,
        // ignore any invalid values to preserve backwards-compatibility
        if (contentEncoding !== 'identity') {
          debug('ignoring unrecognized Content-Encoding ' + contentEncoding)
        }
        dataStream = response
      }
    } else {
      dataStream = response
    }

    if (self.encoding) {
      if (self.dests.length !== 0) {
        console.error('Ignoring encoding parameter as this stream is being piped to another stream which makes the encoding option invalid.')
      } else if (dataStream.setEncoding) {
        dataStream.setEncoding(self.encoding)
      } else {
        // Should only occur on node pre-v0.9.4 (joyent/node@9b5abe5) with
        // zlib streams.
        // If/When support for 0.9.4 is dropped, this should be unnecessary.
        dataStream = dataStream.pipe(stringstream(self.encoding))
      }
    }

    self.emit('response', response)

    self.dests.forEach(function (dest) {
      self.pipeDest(dest)
    })

    dataStream.on('data', function (chunk) {
      self._destdata = true
      self.emit('data', chunk)
    })
    dataStream.on('end', function (chunk) {
      self.emit('end', chunk)
    })
    dataStream.on('error', function (error) {
      self.emit('error', error)
    })
    dataStream.on('close', function () {self.emit('close')})

    if (self.callback) {
      var buffer = bl()
        , strings = []

      self.on('data', function (chunk) {
        if (Buffer.isBuffer(chunk)) {
          buffer.append(chunk)
        } else {
          strings.push(chunk)
        }
      })
      self.on('end', function () {
        debug('end event', self.uri.href)
        if (self._aborted) {
          debug('aborted', self.uri.href)
          return
        }

        if (buffer.length) {
          debug('has body', self.uri.href, buffer.length)
          if (self.encoding === null) {
            // response.body = buffer
            // can't move to this until https://github.com/rvagg/bl/issues/13
            response.body = buffer.slice()
          } else {
            response.body = buffer.toString(self.encoding)
          }
        } else if (strings.length) {
          // The UTF8 BOM [0xEF,0xBB,0xBF] is converted to [0xFE,0xFF] in the JS UTC16/UCS2 representation.
          // Strip this value out when the encoding is set to 'utf8', as upstream consumers won't expect it and it breaks JSON.parse().
          if (self.encoding === 'utf8' && strings[0].length > 0 && strings[0][0] === '\uFEFF') {
            strings[0] = strings[0].substring(1)
          }
          response.body = strings.join('')
        }

        if (self._json) {
          try {
            response.body = JSON.parse(response.body, self._jsonReviver)
          } catch (e) {}
        }
        debug('emitting complete', self.uri.href)
        if(typeof response.body === 'undefined' && !self._json) {
          response.body = self.encoding === null ? new Buffer(0) : ''
        }
        self.emit('complete', response, response.body)
      })
    }
    //if no callback
    else{
      self.on('end', function () {
        if (self._aborted) {
          debug('aborted', self.uri.href)
          return
        }
        self.emit('complete', response)
      })
    }
  }
  debug('finish init function', self.uri.href)
}

Request.prototype.abort = function () {
  var self = this
  self._aborted = true

  if (self.req) {
    self.req.abort()
  }
  else if (self.response) {
    self.response.abort()
  }

  self.emit('abort')
}

Request.prototype.pipeDest = function (dest) {
  var self = this
  var response = self.response
  // Called after the response is received
  if (dest.headers && !dest.headersSent) {
    if (response.caseless.has('content-type')) {
      var ctname = response.caseless.has('content-type')
      if (dest.setHeader) {
        dest.setHeader(ctname, response.headers[ctname])
      }
      else {
        dest.headers[ctname] = response.headers[ctname]
      }
    }

    if (response.caseless.has('content-length')) {
      var clname = response.caseless.has('content-length')
      if (dest.setHeader) {
        dest.setHeader(clname, response.headers[clname])
      } else {
        dest.headers[clname] = response.headers[clname]
      }
    }
  }
  if (dest.setHeader && !dest.headersSent) {
    for (var i in response.headers) {
      // If the response content is being decoded, the Content-Encoding header
      // of the response doesn't represent the piped content, so don't pass it.
      if (!self.gzip || i !== 'content-encoding') {
        dest.setHeader(i, response.headers[i])
      }
    }
    dest.statusCode = response.statusCode
  }
  if (self.pipefilter) {
    self.pipefilter(response, dest)
  }
}

Request.prototype.qs = function (q, clobber) {
  var self = this
  var base
  if (!clobber && self.uri.query) {
    base = self.qsLib.parse(self.uri.query)
  } else {
    base = {}
  }

  for (var i in q) {
    base[i] = q[i]
  }

  if (self.qsLib.stringify(base) === ''){
    return self
  }

  var qs = self.qsLib.stringify(base)

  self.uri = url.parse(self.uri.href.split('?')[0] + '?' + rfc3986(qs))
  self.url = self.uri
  self.path = self.uri.path

  return self
}
Request.prototype.form = function (form) {
  var self = this
  if (form) {
    self.setHeader('content-type', 'application/x-www-form-urlencoded')
    self.body = (typeof form === 'string') ? form.toString('utf8') : self.qsLib.stringify(form).toString('utf8')
    self.body = rfc3986(self.body)
    return self
  }
  // create form-data object
  self._form = new FormData()
  self._form.on('error',function(err) {
    err.message = 'form-data: ' + err.message
    self.emit('error', err)
    self.abort()
  })
  return self._form
}
Request.prototype.multipart = function (multipart) {
  var self = this

  var chunked = false
  var _multipart = multipart.data || multipart

  if (!_multipart.forEach) {
    throw new Error('Argument error, options.multipart.')
  }

  if (self.getHeader('transfer-encoding') === 'chunked') {
    chunked = true
  }
  if (multipart.chunked !== undefined) {
    chunked = multipart.chunked
  }
  if (!chunked) {
    _multipart.forEach(function (part) {
      if(typeof part.body === 'undefined') {
        throw new Error('Body attribute missing in multipart.')
      }
      if (isstream(part.body)) {
        chunked = true
      }
    })
  }

  if (chunked && !self.hasHeader('transfer-encoding')) {
    self.setHeader('transfer-encoding', 'chunked')
  }

  var headerName = self.hasHeader('content-type')
  if (!headerName || self.headers[headerName].indexOf('multipart') === -1) {
    self.setHeader('content-type', 'multipart/related; boundary=' + self.boundary)
  } else {
    self.setHeader(headerName, self.headers[headerName].split(';')[0] + '; boundary=' + self.boundary)
  }

  var parts = chunked ? new CombinedStream() : []
  function add (part) {
    return chunked ? parts.append(part) : parts.push(new Buffer(part))
  }

  if (self.preambleCRLF) {
    add('\r\n')
  }

  _multipart.forEach(function (part) {
    var body = part.body
    var preamble = '--' + self.boundary + '\r\n'
    Object.keys(part).forEach(function (key) {
      if (key === 'body') { return }
      preamble += key + ': ' + part[key] + '\r\n'
    })
    preamble += '\r\n'
    add(preamble)
    add(body)
    add('\r\n')
  })
  add('--' + self.boundary + '--')

  if (self.postambleCRLF) {
    add('\r\n')
  }

  self[chunked ? '_multipart' : 'body'] = parts
  return self
}
Request.prototype.json = function (val) {
  var self = this

  if (!self.hasHeader('accept')) {
    self.setHeader('accept', 'application/json')
  }

  self._json = true
  if (typeof val === 'boolean') {
    if (self.body !== undefined) {
      if (!/^application\/x-www-form-urlencoded\b/.test(self.getHeader('content-type'))) {
        self.body = safeStringify(self.body)
      } else {
        self.body = rfc3986(self.body)
      }
      if (!self.hasHeader('content-type')) {
        self.setHeader('content-type', 'application/json')
      }
    }
  } else {
    self.body = safeStringify(val)
    if (!self.hasHeader('content-type')) {
      self.setHeader('content-type', 'application/json')
    }
  }

  if (typeof self.jsonReviver === 'function') {
    self._jsonReviver = self.jsonReviver
  }

  return self
}
Request.prototype.getHeader = function (name, headers) {
  var self = this
  var result, re, match
  if (!headers) {
    headers = self.headers
  }
  Object.keys(headers).forEach(function (key) {
    if (key.length !== name.length) {
      return
    }
    re = new RegExp(name, 'i')
    match = key.match(re)
    if (match) {
      result = headers[key]
    }
  })
  return result
}
var getHeader = Request.prototype.getHeader

Request.prototype.auth = function (user, pass, sendImmediately, bearer) {
  var self = this

  var authHeader
  if (bearer !== undefined) {
    authHeader = self._auth.bearer(bearer, sendImmediately)
  } else {
    authHeader = self._auth.basic(user, pass, sendImmediately)
  }
  if (authHeader) {
    self.setHeader('authorization', authHeader)
  }

  return self
}

Request.prototype.aws = function (opts, now) {
  var self = this

  if (!now) {
    self._aws = opts
    return self
  }
  var date = new Date()
  self.setHeader('date', date.toUTCString())
  var auth =
    { key: opts.key
    , secret: opts.secret
    , verb: self.method.toUpperCase()
    , date: date
    , contentType: self.getHeader('content-type') || ''
    , md5: self.getHeader('content-md5') || ''
    , amazonHeaders: aws.canonicalizeHeaders(self.headers)
    }
  var path = self.uri.path
  if (opts.bucket && path) {
    auth.resource = '/' + opts.bucket + path
  } else if (opts.bucket && !path) {
    auth.resource = '/' + opts.bucket
  } else if (!opts.bucket && path) {
    auth.resource = path
  } else if (!opts.bucket && !path) {
    auth.resource = '/'
  }
  auth.resource = aws.canonicalizeResource(auth.resource)
  self.setHeader('authorization', aws.authorization(auth))

  return self
}
Request.prototype.httpSignature = function (opts) {
  var self = this
  httpSignature.signRequest({
    getHeader: function(header) {
      return getHeader(header, self.headers)
    },
    setHeader: function(header, value) {
      self.setHeader(header, value)
    },
    method: self.method,
    path: self.path
  }, opts)
  debug('httpSignature authorization', self.getHeader('authorization'))

  return self
}

Request.prototype.hawk = function (opts) {
  var self = this
  self.setHeader('Authorization', hawk.client.header(self.uri, self.method, opts).field)
}

Request.prototype.oauth = function (_oauth) {
  var self = this

  var result = oauth.oauth({
    uri: self.uri,
    method: self.method,
    headers: self.headers,
    body: self.body,
    oauth: _oauth,
    qsLib: self.qsLib
  })

  if (result.transport === 'header') {
    self.setHeader('Authorization', result.oauth)
  }
  else if (result.transport === 'query') {
    self.path += result.oauth
  }
  else if (result.transport === 'body') {
    self.body = result.oauth
  }

  return self
}

Request.prototype.jar = function (jar) {
  var self = this
  var cookies

  if (self._redirectsFollowed === 0) {
    self.originalCookieHeader = self.getHeader('cookie')
  }

  if (!jar) {
    // disable cookies
    cookies = false
    self._disableCookies = true
  } else {
    var targetCookieJar = (jar && jar.getCookieString) ? jar : globalCookieJar
    var urihref = self.uri.href
    //fetch cookie in the Specified host
    if (targetCookieJar) {
      cookies = targetCookieJar.getCookieString(urihref)
    }
  }

  //if need cookie and cookie is not empty
  if (cookies && cookies.length) {
    if (self.originalCookieHeader) {
      // Don't overwrite existing Cookie header
      self.setHeader('cookie', self.originalCookieHeader + '; ' + cookies)
    } else {
      self.setHeader('cookie', cookies)
    }
  }
  self._jar = jar
  return self
}


// Stream API
Request.prototype.pipe = function (dest, opts) {
  var self = this

  if (self.response) {
    if (self._destdata) {
      throw new Error('You cannot pipe after data has been emitted from the response.')
    } else if (self._ended) {
      throw new Error('You cannot pipe after the response has been ended.')
    } else {
      stream.Stream.prototype.pipe.call(self, dest, opts)
      self.pipeDest(dest)
      return dest
    }
  } else {
    self.dests.push(dest)
    stream.Stream.prototype.pipe.call(self, dest, opts)
    return dest
  }
}
Request.prototype.write = function () {
  var self = this
  if (!self._started) {
    self.start()
  }
  return self.req.write.apply(self.req, arguments)
}
Request.prototype.end = function (chunk) {
  var self = this
  if (chunk) {
    self.write(chunk)
  }
  if (!self._started) {
    self.start()
  }
  self.req.end()
}
Request.prototype.pause = function () {
  var self = this
  if (!self.response) {
    self._paused = true
  } else {
    self.response.pause.apply(self.response, arguments)
  }
}
Request.prototype.resume = function () {
  var self = this
  if (!self.response) {
    self._paused = false
  } else {
    self.response.resume.apply(self.response, arguments)
  }
}
Request.prototype.destroy = function () {
  var self = this
  if (!self._ended) {
    self.end()
  } else if (self.response) {
    self.response.destroy()
  }
}

Request.defaultProxyHeaderWhiteList =
  defaultProxyHeaderWhiteList.slice()

Request.defaultProxyHeaderExclusiveList =
  defaultProxyHeaderExclusiveList.slice()

// Exports

Request.prototype.toJSON = requestToJSON
module.exports = Request
