#! /usr/bin/env node

var path = require('path'),
    fs = require('fs'),
    url = require('url'),
    mime = require('mime'),
    showDir = require('./ecstatic/showdir'),
    version = JSON.parse(
      fs.readFileSync(__dirname + '/../package.json').toString()
    ).version,
    status = require('./ecstatic/status-handlers'),
    etag = require('./ecstatic/etag'),
    optsParser = require('./ecstatic/opts');

var ecstatic = module.exports = function (dir, options) {
  if (typeof dir !== 'string') {
    options = dir;
    dir = options.root;
  }

  var root = path.join(path.resolve(dir), '/'),
      opts = optsParser(options),
      cache = opts.cache,
      autoIndex = opts.autoIndex,
      baseDir = opts.baseDir,
      defaultExt = opts.defaultExt,
      handleError = opts.handleError;

  opts.root = dir;

  return function middleware (req, res, next) {

    // Strip any null bytes from the url
    while(req.url.indexOf('%00') !== -1) {
      req.url = req.url.replace(/\%00/g, '');
    }
    // Figure out the path for the file from the given url
    var parsed = url.parse(req.url);
    try {
      decodeURI(req.url); // check validity of url
      var pathname = decodeURI(parsed.pathname);
    }
    catch (err) {
      return status[400](res, next, { error: err });
    }

    var file = path.normalize(
          path.join(root,
            path.relative(
              path.join('/', baseDir),
              pathname
            )
          )
        ),
        gzipped = file + '.gz';

    // Set common headers.
    res.setHeader('server', 'ecstatic-'+version);

    // TODO: This check is broken, which causes the 403 on the
    // expected 404.
    if (file.slice(0, root.length) !== root) {
      return status[403](res, next);
    }

    if (req.method && (req.method !== 'GET' && req.method !== 'HEAD' )) {
      return status[405](res, next);
    }

    // Look for a gzipped file if this is turned on
    if (opts.gzip && shouldCompress(req)) {
      fs.stat(gzipped, function (err, stat) {
        if (!err && stat.isFile()) {
          file = gzipped;
          return serve(stat);
        }
      });
    }

    fs.stat(file, function (err, stat) {
      if (err && err.code === 'ENOENT') {
        if (req.statusCode == 404) {
          // This means we're already trying ./404.html
          status[404](res, next);
        }
        else if (defaultExt && !path.extname(parsed.pathname).length) {
          //
          // If no file extension is specified and there is a default extension
          // try that before rendering 404.html.
          //
          middleware({
            url: parsed.pathname + '.' + defaultExt + ((parsed.search)? parsed.search:'')
          }, res, next);
        }
        else {
          // Try for ./404.html
          middleware({
            url: (handleError ? ('/' + path.join(baseDir, '404.html')) : req.url),
            statusCode: 404 // Override the response status code
          }, res, next);
        }
      }
      else if (err) {
        status[500](res, next, { error: err });
      }
      else if (stat.isDirectory()) {
        // 302 to / if necessary
        if (!parsed.pathname.match(/\/$/)) {
          res.statusCode = 302;
          res.setHeader('location', parsed.pathname + '/' +
            (parsed.query? ('?' + parsed.query):'')
          );
          return res.end();
        }

        if (autoIndex) {
          return middleware({
            url: path.join(pathname, '/index.html')
          }, res, function (err) {
            if (err) {
              return status[500](res, next, { error: err });
            }
            if (opts.showDir) {
              return showDir(opts, stat)(req, res);
            }

            return status[403](res, next);
          });
        }

        if (opts.showDir) {
          return showDir(opts, stat)(req, res);
        }

        status[404](res, next);

      }
      else {
        serve(stat);
      }
    });

    function serve(stat) {

      // TODO: Helper for this, with default headers.
      res.setHeader('etag', etag(stat));
      res.setHeader('last-modified', (new Date(stat.mtime)).toUTCString());
      res.setHeader('cache-control', cache);

      // Return a 304 if necessary
      if ( req.headers
        && (
          (req.headers['if-none-match'] === etag(stat))
          || (new Date(Date.parse(req.headers['if-modified-since'])) >= stat.mtime)
        )
      ) {
        return status[304](res, next);
      }

      res.setHeader('content-length', stat.size);

      // Do a MIME lookup, fall back to octet-stream and handle gzip
      // special case.
      var contentType = mime.lookup(file), charSet;

      if (contentType) {
        charSet = mime.charsets.lookup(contentType);
        if (charSet) {
          contentType += '; charset=' + charSet;
        }
      }

      if (path.extname(file) === '.gz') {
        res.setHeader('Content-Encoding', 'gzip');

        // strip gz ending and lookup mime type
        contentType = mime.lookup(path.basename(file, ".gz"));
      }

      res.setHeader('content-type', contentType || 'application/octet-stream');

      if (req.method === "HEAD") {
        res.statusCode = req.statusCode || 200; // overridden for 404's
        return res.end();
      }

      var stream = fs.createReadStream(file);

      stream.pipe(res);
      stream.on('error', function (err) {
        status['500'](res, next, { error: err });
      });

      stream.on('end', function () {
        res.statusCode = 200;
        res.end();
      });
    }
  };
};

ecstatic.version = version;
ecstatic.showDir = showDir;

// Check to see if we should try to compress a file with gzip.
function shouldCompress(req) {
  var headers = req.headers;

  return headers && headers['accept-encoding'] &&
    headers['accept-encoding']
      .split(",")
      .some(function (el) {
        return ['*','compress', 'gzip', 'deflate'].indexOf(el) != -1;
      })
    ;
}

if(!module.parent) {
  var http = require('http'),
      opts = require('optimist').argv,
      port = opts.port || opts.p || 8000,
      dir = opts.root || opts._[0] || process.cwd();

  if(opts.help || opts.h) {
    var u = console.error;
    u('usage: ecstatic [dir] {options} --port PORT');
    u('see https://npm.im/ecstatic for more docs');
    return;
  }

  http.createServer(ecstatic(dir, opts))
    .listen(port, function () {
      console.log('ecstatic serving ' + dir + ' on port ' + port);
    });
}
