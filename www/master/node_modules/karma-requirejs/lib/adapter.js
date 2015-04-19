(function(karma, requirejs, locationPathname) {

// monkey patch requirejs, to use append timestamps to sources
// to take advantage of karma's heavy caching
// it would work even without this hack, but with reloading all the files all the time

var normalizePath = function(path) {
  var normalized = [];
  var parts = path.split('/');

  for (var i = 0; i < parts.length; i++) {
    if (parts[i] === '.') {
      continue;
    }

    if (parts[i] === '..' && normalized.length && normalized[normalized.length - 1] !== '..') {
      normalized.pop();
      continue;
    }

    normalized.push(parts[i]);
  }

  return normalized.join('/');
};

var createPatchedLoad = function(files, originalLoadFn, locationPathname) {
  var IS_DEBUG = /debug\.html$/.test(locationPathname);

  return function(context, moduleName, url) {
    url = normalizePath(url);

    if (files.hasOwnProperty(url)) {
      if (!IS_DEBUG) {
        url = url + '?' + files[url];
      }
    } else {
      if (!/https?:\/\/\S+\.\S+/i.test(url)) {
        console.error('There is no timestamp for ' + url + '!');
      }
    }

    return originalLoadFn.call(this, context, moduleName, url);
  };
};

// make it async
karma.loaded = function() {};

// patch require.js
requirejs.load = createPatchedLoad(karma.files, requirejs.load, locationPathname);

})(window.__karma__, window.requirejs, window.location.pathname);
