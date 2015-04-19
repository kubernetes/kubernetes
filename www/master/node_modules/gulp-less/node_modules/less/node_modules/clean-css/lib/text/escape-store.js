module.exports = function EscapeStore(placeholderRoot) {
  placeholderRoot = 'ESCAPED_' + placeholderRoot + '_CLEAN_CSS';

  var placeholderToData = {};
  var dataToPlaceholder = {};
  var count = 0;
  var nextPlaceholder = function() {
    return '__' + placeholderRoot + (count++) + '__';
  };
  var pattern = '(__' + placeholderRoot + '\\d{1,}__)';

  return {
    placeholderPattern: pattern,

    placeholderRegExp: new RegExp(pattern, 'g'),

    store: function(data) {
      var placeholder = dataToPlaceholder[data];
      if (!placeholder) {
        placeholder = nextPlaceholder();
        placeholderToData[placeholder] = data;
        dataToPlaceholder[data] = placeholder;
      }

      return placeholder;
    },

    restore: function(placeholder) {
      return placeholderToData[placeholder];
    }
  };
};
