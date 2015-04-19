module.exports = function EmptyRemoval(data) {
  var stripEmpty = function(cssData) {
    var tempData = [];
    var nextEmpty = 0;
    var cursor = 0;

    for (; nextEmpty < cssData.length;) {
      nextEmpty = cssData.indexOf('{}', cursor);
      if (nextEmpty == -1)
        break;

      var startsAt = nextEmpty - 1;
      while (cssData[startsAt] && cssData[startsAt] != '}' && cssData[startsAt] != '{' && cssData[startsAt] != ';')
        startsAt--;

      tempData.push(cssData.substring(cursor, startsAt + 1));
      cursor = nextEmpty + 2;
    }

    return tempData.length > 0 ?
      stripEmpty(tempData.join('') + cssData.substring(cursor, cssData.length)) :
      cssData;
  };

  return {
    process: function() {
      return stripEmpty(data);
    }
  };
};
