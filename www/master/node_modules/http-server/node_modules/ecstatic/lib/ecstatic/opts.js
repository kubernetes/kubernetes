// This is so you can have options aliasing and defaults in one place.

module.exports = function (opts) {

  var autoIndex = true,
      showDir = false,
      humanReadable = true,
      si = false,
      cache = 'max-age=3600',
      gzip = false,
      defaultExt,
      handleError = true;

  if (opts) {
    [
      'autoIndex',
      'autoindex'
    ].some(function (k) {
      if (typeof opts[k] !== 'undefined' && opts[k] !== null) {
        autoIndex = opts[k];
        return true;
      }
    });

    [
      'showDir',
      'showdir'
    ].some(function (k) {
      if (typeof opts[k] !== 'undefined' && opts[k] !== null) {
        showDir = opts[k];
        return true;
      }
    });

    [
      'humanReadable',
      'humanreadable',
      'human-readable'
    ].some(function (k) {
      if (typeof opts[k] !== 'undefined' && opts[k] !== null) {
        humanReadable = opts[k];
        return true;
      }
    });

    [
      'si',
      'index'
    ].some(function (k) {
      if (typeof opts[k] !== 'undefined' && opts[k] !== null) {
        si = opts[k];
        return true;
      }
    });

    if (opts.defaultExt) {
      if (opts.defaultExt === true) {
        defaultExt = 'html';
      }
      else {
        defaultExt = opts.defaultExt;
      }
    }

    if (typeof opts.cache !== 'undefined' && opts.cache !== null) {
      if (typeof opts.cache === 'string') {
        cache = opts.cache;
      }
      else if(typeof opts.cache === 'number') {
        cache = 'max-age=' + opts.cache;
      }
    }

    if (typeof opts.gzip !== 'undefined' && opts.gzip !== null) {
      gzip = opts.gzip;
    }

    [
      'handleError',
      'handleerror'
    ].some(function (k) {
        if (typeof opts[k] !== 'undefined' && opts[k] !== null) {
            handleError = opts[k];
            return true;
        }
    });
  }

  return {
    cache: cache,
    autoIndex: autoIndex,
    showDir: showDir,
    humanReadable: humanReadable,
    si: si,
    defaultExt: defaultExt,
    baseDir: (opts && opts.baseDir) || '/',
    gzip: gzip,
    handleError: handleError
  };
};
