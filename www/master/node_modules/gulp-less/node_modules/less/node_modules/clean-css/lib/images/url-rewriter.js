var path = require('path');
var url = require('url');

module.exports = {
  process: function(data, options) {
    var tempData = [];
    var nextStart = 0;
    var nextEnd = 0;
    var cursor = 0;

    for (; nextEnd < data.length;) {
      nextStart = data.indexOf('url(', nextEnd);
      if (nextStart == -1)
        break;

      nextEnd = data.indexOf(')', nextStart + 4);
      if (nextEnd == -1)
        break;

      tempData.push(data.substring(cursor, nextStart));
      var url = data.substring(nextStart + 4, nextEnd);
      if (!/\/\*|\*\//.test(url))
        url = url.replace(/['"]/g, '');

      tempData.push('url(' + this._rebased(url, options) + ')');
      cursor = nextEnd + 1;
    }

    return tempData.length > 0 ?
      tempData.join('') + data.substring(cursor, data.length) :
      data;
  },

  _rebased: function(resource, options) {
    var specialUrl = resource[0] == '/' ||
      resource[0] == '#' ||
      resource.substring(resource.length - 4) == '.css' ||
      resource.indexOf('data:') === 0 ||
      /^https?:\/\//.exec(resource) !== null ||
      /__\w+__/.exec(resource) !== null;
    var rebased;

    if (specialUrl)
      return resource;

    if (/https?:\/\//.test(options.toBase))
      return url.resolve(options.toBase, resource);

    if (options.absolute) {
      rebased = path
        .resolve(path.join(options.fromBase, resource))
        .replace(options.toBase, '');
    } else {
      rebased = path.relative(options.toBase, path.join(options.fromBase, resource));
    }

    return process.platform == 'win32' ?
      rebased.replace(/\\/g, '/') :
      rebased;
  }
};
