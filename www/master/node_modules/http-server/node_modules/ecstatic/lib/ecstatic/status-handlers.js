// not modified
exports['304'] = function (res, next) {
  res.statusCode = 304;
  res.end();
};

// access denied
exports['403'] = function (res, next) {
  res.statusCode = 403;
  if (typeof next === "function") {
    next();
  }
  else {
    if (res.writable) {
      res.setHeader('content-type', 'text/plain');
      res.end('ACCESS DENIED');
    }
  }
};

// disallowed method
exports['405'] = function (res, next, opts) {
  res.statusCode = 405;
  if (typeof next === "function") {
    next();
  }
  else {
    res.setHeader('allow', (opts && opts.allow) || 'GET, HEAD');
    res.end();    
  }
};

// not found
exports['404'] = function (res, next) {
  res.statusCode = 404;
  if (typeof next === "function") {
    next();
  }
  else {
    if (res.writable) {
      res.setHeader('content-type', 'text/plain');
      res.end('File not found. :(');
    }
  }
};

// flagrant error
exports['500'] = function (res, next, opts) {
  res.statusCode = 500;
  res.end(opts.error.stack || opts.error.toString() || "No specified error");
};

// bad request
exports['400'] = function (res, next, opts) {
  res.statusCode = 400;
  res.end(opts && opts.error ? String(opts.error) : 'Malformed request.');
};
