var EscapeStore = require('./escape-store');

module.exports = function Urls(context) {
  var urls = new EscapeStore('URL');

  return {
    // Strip urls by replacing them by a special
    // marker for further restoring. It's done via string scanning
    // instead of regexps to speed up the process.
    escape: function(data) {
      var nextStart = 0;
      var nextEnd = 0;
      var cursor = 0;
      var tempData = [];

      for (; nextEnd < data.length;) {
        nextStart = data.indexOf('url(', nextEnd);
        if (nextStart == -1)
          break;

        nextEnd = data.indexOf(')', nextStart);
        // Following lines are a safety mechanism to ensure
        // incorrectly terminated urls are processed correctly.
        if (nextEnd == -1) {
          nextEnd = data.indexOf('}', nextStart);

          if (nextEnd == -1)
            nextEnd = data.length;
          else
            nextEnd--;

          context.warnings.push('Broken URL declaration: \'' + data.substring(nextStart, nextEnd + 1) + '\'.');
        }

        var url = data.substring(nextStart, nextEnd + 1);
        var placeholder = urls.store(url);
        tempData.push(data.substring(cursor, nextStart));
        tempData.push(placeholder);
        cursor = nextEnd + 1;
      }

      return tempData.length > 0 ?
        tempData.join('') + data.substring(cursor, data.length) :
        data;
    },

    restore: function(data) {
      return data.replace(urls.placeholderRegExp, function(placeholder) {
        return urls.restore(placeholder).replace(/\s/g, '');
      });
    }
  };
};
